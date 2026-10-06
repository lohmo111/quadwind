"""Ветровая лаборатория: интерактивная симуляция с ползунками.

Ветер задаётся по компасу (откуда дует: С, СВ, В, …), силой, турбулентностью и сидом.
Можно менять массу, ослабить мотор, добавить шум и задержку, вызвать порыв или толчок.
Все включённые регуляторы летят в одном и том же ветре и с одним и тем же шумом датчиков.

Запуск:
    python scripts/wind_lab.py                                   # PID и PD
    python scripts/wind_lab.py --run runs/pilot2/rppo/seed_0 --run runs/pilot2/ppo_mlp/seed_0
    python scripts/wind_lab.py --snapshot lab.png --seconds 12   # без окна: сохранить картинку
    python scripts/wind_lab.py --gif lab.gif --seconds 15 --gust-at 6 --from-deg 45 --speed 3   # анимация
Карта — вид сверху: x — восток, y — север, цель в центре.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from quadwind.conditions import NOMINAL_SPACE  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig, HoverCore  # noqa: E402
from quadwind.params import CF2X  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
from quadwind import style  # noqa: E402
from quadwind.wind import COMPASS, LiveWind  # noqa: E402

LAB_SECONDS = 600.0          # длина «эпизода» в лаборатории (потом незаметный перезапуск)
HISTORY_S = 15.0             # сколько секунд показывать на графиках
TRAIL_S = 6.0                # длина следа на карте
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
PRETTY = {"rppo": "RPPO", "ppo_mlp": "MLP", "ppo_mlp_noprev": "MLP без a", "ppo_stack4": "MLP-k4",
          "ppo_stack16": "MLP-k16", "ppo_oracle": "Оракул", "ppo_mlp_integral": "MLP+∫", "rppo_calm": "RPPO-штиль"}
NAMES_RU = {"С": "северный", "СВ": "северо-восточный", "В": "восточный", "ЮВ": "юго-восточный",
            "Ю": "южный", "ЮЗ": "юго-западный", "З": "западный", "СЗ": "северо-западный"}


def compass_name(from_deg: float) -> str:
    label = min(COMPASS, key=lambda k: abs((COMPASS[k] - from_deg + 180) % 360 - 180))
    return label


class Controller:
    """Один регулятор со своей копией среды (N = 1); ветер и шум общие для всех."""

    def __init__(self, name: str, policy, env_cfg: EnvConfig, color: str):
        self.name, self.policy, self.color = name, policy, color
        self.cfg = replace(env_cfg, episode_seconds=LAB_SECONDS)
        self.core = HoverCore(self.cfg, 1)
        self.enabled = True
        self.crashes = 0
        self.crash_timer = 0
        self.clear_history()

    def clear_history(self) -> None:
        self.t, self.err, self.tilt, self.xy = [], [], [], []

    def reset(self, seed: int, noise_scale: float, act_delay: int, mass_ratio: float, motor1: float) -> None:
        cond = NOMINAL_SPACE.sample(np.random.default_rng(seed), seed=(0x1AB, seed))
        cond = replace(cond, noise_scale=noise_scale, act_delay=act_delay)
        self.core.reset_envs([0], [cond])
        self.core.dyn.mass[0] = mass_ratio * CF2X.mass
        self.core.dyn.motor_eff[0, 0] = motor1
        self.policy.reset(1)
        self.start = np.ones(1, dtype=bool)

    def step(self, wind: np.ndarray, t: float) -> bool:
        core = self.core
        core.wind_override = wind
        action = self.policy.act(core.observe(), self.start)
        self.start[:] = False
        _, terminated, truncated = core.step(action)
        e = core.dyn.pos[0] - core.target
        self.t.append(t)
        self.err.append(e.copy())
        self.tilt.append(float(np.degrees(np.arccos(np.clip(core.dyn.R[0, 2, 2], -1, 1)))))
        self.xy.append(e[:2].copy())
        return bool(terminated[0] or truncated[0])

    def trim(self, t_now: float) -> None:
        k = 0
        while k < len(self.t) and self.t[k] < t_now - HISTORY_S:
            k += 1
        if k:
            del self.t[:k], self.err[:k], self.tilt[:k], self.xy[:k]


class Lab:
    def __init__(self, controllers: list[Controller]):
        self.controllers = controllers
        self.dt = controllers[0].core.dt
        self.wind = LiveWind(self.dt)
        self.seed = 0
        self.noise_scale, self.act_delay, self.mass_ratio, self.motor1 = 1.0, 1, 1.0, 1.0
        self.t = 0.0
        self.wind_log: list[tuple[float, np.ndarray]] = []
        self.reset()

    def reset(self) -> None:
        self.wind.reseed(self.seed)
        self.t = 0.0
        self.wind_log.clear()
        for c in self.controllers:
            c.reset(self.seed, self.noise_scale, self.act_delay, self.mass_ratio, self.motor1)
            c.clear_history()
            c.crash_timer = 0

    def set_mass(self, ratio: float) -> None:
        self.mass_ratio = ratio
        for c in self.controllers:
            c.core.dyn.mass[0] = ratio * CF2X.mass

    def set_motor1(self, eff: float) -> None:
        self.motor1 = eff
        for c in self.controllers:
            c.core.dyn.motor_eff[0, 0] = eff

    def set_delay(self, steps: int) -> None:
        self.act_delay = steps
        for c in self.controllers:
            c.core.act_delay[0] = steps

    def set_noise(self, scale: float) -> None:
        """Шум перегенерируется из сида, одинаково для всех регуляторов."""
        self.noise_scale = scale
        for c in self.controllers:
            core = c.core
            rng = np.random.default_rng([0x5E, self.seed])
            core.noise[0] = rng.standard_normal(core.noise.shape[1:]) * core.noise_std * scale

    def push(self, force_g: float = 0.3, duration: float = 0.3) -> None:
        """Толчок поперёк ветра (в долях веса)."""
        th = np.radians(self.wind.from_deg + 90.0)
        f = force_g * CF2X.weight * np.array([np.sin(th), np.cos(th), 0.0])
        for c in self.controllers:
            core = c.core
            core.ev_push_t[0] = core.step_count[0] * core.dt
            core.ev_push_dur[0] = duration
            core.ev_push_force[0] = f

    def advance(self, n_steps: int) -> None:
        for _ in range(n_steps):
            w = self.wind.step()
            self.t += self.dt
            self.wind_log.append((self.t, w))
            for c in self.controllers:
                if not c.enabled:
                    continue
                if c.crash_timer > 0:                      # после падения пауза 1 с, затем снова в точке
                    c.crash_timer -= 1
                    if c.crash_timer == 0:
                        c.reset(self.seed, self.noise_scale, self.act_delay, self.mass_ratio, self.motor1)
                    continue
                if c.step(w, self.t):
                    if c.core.step_count[0] < c.core.max_steps:
                        c.crashes += 1
                        c.crash_timer = int(1.0 / self.dt)
                    else:
                        c.reset(self.seed, self.noise_scale, self.act_delay, self.mass_ratio, self.motor1)
        while self.wind_log and self.wind_log[0][0] < self.t - HISTORY_S:
            self.wind_log.pop(0)
        for c in self.controllers:
            c.trim(self.t)


# ------------------------------------------------------------------------------------------- GUI
MAP_LEVELS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)           # м, полуширина карты
ERR_LEVELS = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0)  # см
WIND_LEVELS = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0)             # м/с
WINDOW_S, SCROLL_S = 20.0, 5.0                              # окно графиков и шаг его прокрутки, с


def level(x: float, levels) -> float:
    return next((v for v in levels if v >= x), levels[-1])


class LabView:
    """Окно лаборатории. Быстрая анимация через блиттинг: статичная часть (ползунки, оси, подписи)
    рисуется редко, каждый кадр перерисовываются только движущиеся элементы. Пределы осей меняются
    ступеньками, поэтому полная перерисовка нужна лишь изредка."""

    def __init__(self, lab: Lab, interactive: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, Slider

        self.lab, self.plt, self.interactive = lab, plt, interactive
        self.paused = False
        self.sim_speed = 1.0
        self.limits = None
        self.background = None
        self.frame_times: list[float] = []
        self._last_wall = None
        self._sim_debt = 0.0
        fig = plt.figure(figsize=(14, 8.4))
        if interactive and fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("Ветровая лаборатория — quadwind")
        self.fig = fig
        self.ax_map = fig.add_axes([0.04, 0.42, 0.40, 0.55])
        self.ax_err = fig.add_axes([0.50, 0.73, 0.47, 0.23])
        self.ax_wind = fig.add_axes([0.50, 0.44, 0.47, 0.22], sharex=self.ax_err)
        self._setup_map()
        self._setup_plots()

        # ---- ползунки
        def slider(row, label, vmin, vmax, init, step=None, col=0):
            x = 0.12 if col == 0 else 0.53
            ax = fig.add_axes([x, 0.335 - row * 0.042, 0.19, 0.025])
            return Slider(ax, label, vmin, vmax, valinit=init, valstep=step)

        w = lab.wind
        self.s_speed = slider(0, "Ветер, м/с", 0.0, 8.0, w.speed, 0.1)
        self.s_dir = slider(1, "Откуда дует, °", 0.0, 359.0, w.from_deg, 1.0)
        self.s_turb = slider(2, "Турбулентность σ, м/с", 0.0, 1.5, w.turb_sigma, 0.05)
        self.s_vert = slider(3, "Верт. ветер, м/с", -1.0, 1.0, w.vertical, 0.05)
        self.s_seed = slider(4, "Сид", 0, 99, lab.seed, 1)
        self.s_mass = slider(0, "Масса, ×m₀", 0.7, 1.4, lab.mass_ratio, 0.01, col=1)
        self.s_motor = slider(1, "Мотор №1, %", 50, 100, 100 * lab.motor1, 1, col=1)
        self.s_noise = slider(2, "Шум датчиков, ×", 0.0, 3.0, lab.noise_scale, 0.1, col=1)
        self.s_delay = slider(3, "Задержка, ×20 мс", 0, 3, lab.act_delay, 1, col=1)
        self.s_simspeed = slider(4, "Скорость симуляции", 0.25, 4.0, 1.0, 0.25, col=1)
        self.s_speed.on_changed(lambda v: setattr(lab.wind, "speed", float(v)))
        self.s_dir.on_changed(lambda v: setattr(lab.wind, "from_deg", float(v)))
        self.s_turb.on_changed(lambda v: setattr(lab.wind, "turb_sigma", float(v)))
        self.s_vert.on_changed(lambda v: setattr(lab.wind, "vertical", float(v)))
        self.s_seed.on_changed(self._on_seed)
        self.s_mass.on_changed(lambda v: lab.set_mass(float(v)))
        self.s_motor.on_changed(lambda v: lab.set_motor1(float(v) / 100.0))
        self.s_noise.on_changed(lambda v: lab.set_noise(float(v)))
        self.s_delay.on_changed(lambda v: lab.set_delay(int(v)))
        self.s_simspeed.on_changed(lambda v: setattr(self, "sim_speed", float(v)))

        # ---- роза ветров: кнопки «откуда дует»
        self.compass_buttons = []
        grid = {"СЗ": (0, 0), "С": (0, 1), "СВ": (0, 2), "З": (1, 0), "В": (1, 2), "ЮЗ": (2, 0), "Ю": (2, 1),
                "ЮВ": (2, 2)}
        fig.text(0.78, 0.348, "Откуда дует:", fontsize=9)
        for label, (r, c) in grid.items():
            ax = fig.add_axes([0.78 + c * 0.045, 0.30 - r * 0.045, 0.04, 0.038])
            b = Button(ax, label)
            b.on_clicked(lambda _e, d=COMPASS[label]: self.s_dir.set_val(d))
            self.compass_buttons.append(b)
        ax = fig.add_axes([0.78 + 0.045, 0.30 - 0.045, 0.04, 0.038])
        self.b_calm = Button(ax, "штиль")
        self.b_calm.on_clicked(lambda _e: self.s_speed.set_val(0.0))

        # ---- кнопки действий
        def button(i, label, cb):
            ax = fig.add_axes([0.04 + i * 0.085, 0.085, 0.08, 0.045])
            b = Button(ax, label)
            b.on_clicked(cb)
            return b

        self.b_gust = button(0, "Порыв!", lambda _e: lab.wind.gust(3.0, 1.0))
        self.b_push = button(1, "Толчок", lambda _e: lab.push())
        self.b_reset = button(2, "Сброс", lambda _e: self._reset())
        self.b_pause = button(3, "Пауза", lambda _e: self._toggle_pause())

        # ---- какие регуляторы летят
        ax = fig.add_axes([0.40, 0.015, 0.16, 0.115])
        fig.text(0.40, 0.135, "Регуляторы (вкл/выкл):", fontsize=9)
        self.checks = CheckButtons(ax, [c.name for c in lab.controllers], [c.enabled for c in lab.controllers])
        for text, c in zip(self.checks.labels, lab.controllers):
            text.set_color(c.color)
        self.checks.on_clicked(self._on_check)
        self.status = fig.text(0.58, 0.015, "", fontsize=8.5, family="monospace", va="bottom")

        self.animated = [*self.trails, *self.dots, *self.tilt_lines, self.wind_arrow, self.wind_label,
                         *self.err_lines, self.wind_h, self.wind_z, self.status]
        if interactive:
            for artist in self.animated:
                artist.set_animated(True)
            fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.update()

    # ---- оформление
    def _setup_map(self):
        ax = self.ax_map
        ax.set_aspect("equal")
        ax.set_title("Вид сверху (цель — в центре)", fontsize=10)
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.axvline(0, color="#ccc", lw=0.6)
        ax.plot(0, 0, marker="+", color="k", ms=14, mew=1.5)
        self.compass_text = {k: ax.text(0, 0, k, ha="center", va="center", fontsize=13, weight="bold",
                                        color="#555") for k in ("С", "Ю", "З", "В")}
        self.trails = [ax.plot([], [], color=c.color, lw=1.2, alpha=0.7)[0] for c in self.lab.controllers]
        self.dots = [ax.plot([], [], "o", color=c.color, ms=9, mec="k", mew=0.6, label=c.name)[0]
                     for c in self.lab.controllers]
        self.tilt_lines = [ax.plot([], [], color=c.color, lw=2.2)[0] for c in self.lab.controllers]
        self.wind_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle="-|>", color="#0a5", lw=3, mutation_scale=22))
        self.wind_label = ax.text(0, 0, "", ha="center", fontsize=10, color="#0a5")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8, title="линия — куда наклонён", title_fontsize=7)
        ax.set_xlabel("восток →, м")
        ax.set_ylabel("север →, м")

    def _setup_plots(self):
        self.ax_err.set_ylabel("ошибка, см")
        self.ax_err.set_title("Отклонение от точки", fontsize=10)
        self.err_lines = [self.ax_err.plot([], [], color=c.color, lw=1.3)[0] for c in self.lab.controllers]
        self.ax_wind.set_ylabel("ветер, м/с")
        self.ax_wind.set_xlabel("время, с")
        self.wind_h, = self.ax_wind.plot([], [], color="#0a5", lw=1.3, label="горизонтальный |w|")
        self.wind_z, = self.ax_wind.plot([], [], color="#0a5", lw=1.0, ls="--", label="вертикальный w_z")
        self.ax_wind.legend(loc="upper left", fontsize=8)
        for ax in (self.ax_err, self.ax_wind):
            ax.grid(alpha=0.3)

    # ---- обработчики
    def _on_seed(self, v):
        self.lab.seed = int(v)
        self._reset()

    def _reset(self):
        self.lab.reset()
        self.lab.set_noise(self.lab.noise_scale)
        self.limits = None

    def _toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.label.set_text("Продолжить" if self.paused else "Пауза")
        if self.interactive:
            self.fig.canvas.draw_idle()

    def _on_check(self, label):
        for c in self.lab.controllers:
            if c.name == label:
                c.enabled = not c.enabled
                c.clear_history()

    # ---- пределы осей (меняются ступеньками)
    def _limits(self) -> tuple:
        lab, t = self.lab, self.lab.t
        active = [c for c in lab.controllers if c.enabled and c.xy]
        n3 = int(3 / lab.dt)
        need_L = max([0.0] + [1.3 * np.abs(np.array(c.xy[-n3:])).max() for c in active])
        t0 = SCROLL_S * np.floor(max(0.0, t - (WINDOW_S - SCROLL_S)) / SCROLL_S)
        in_win = [(100 * np.linalg.norm(np.array(c.err), axis=1))[np.array(c.t) >= t0].max(initial=0.0)
                  for c in active]
        need_E = 1.15 * max([0.0] + in_win)
        if lab.wind_log:
            ww = np.array([x[1] for x in lab.wind_log])
            need_W = max(np.linalg.norm(ww[:, :2], axis=1).max(), np.abs(ww[:, 2]).max()) + 0.5
        else:
            need_W = 0.0
        if self.limits is None:
            return level(need_L, MAP_LEVELS), t0, level(need_E, ERR_LEVELS), level(need_W, WIND_LEVELS)
        L, old_t0, E, W = self.limits
        L = level(need_L, MAP_LEVELS) if need_L > L or need_L < 0.4 * L else L           # гистерезис
        E = level(need_E, ERR_LEVELS) if need_E > E or (t0 != old_t0 and need_E < 0.35 * E) else E
        W = level(need_W, WIND_LEVELS) if need_W > W or (t0 != old_t0 and need_W < 0.4 * W) else W
        return L, t0, E, W

    def _apply_limits(self, limits) -> None:
        L, t0, E, W = limits
        self.ax_map.set_xlim(-L, L)
        self.ax_map.set_ylim(-L, L)
        for k, (x, y) in {"С": (0, 0.92), "Ю": (0, -0.92), "З": (-0.92, 0), "В": (0.92, 0)}.items():
            self.compass_text[k].set_position((x * L, y * L))
        self.ax_err.set_xlim(t0, t0 + WINDOW_S)
        self.ax_err.set_ylim(0, E)
        self.ax_wind.set_ylim(-min(2.0, W / 2), W)
        self.limits = limits

    # ---- кадр
    def update(self) -> bool:
        """Обновляет данные артистов. True — поменялись пределы осей (нужна полная перерисовка)."""
        lab = self.lab
        limits = self._limits()
        changed = limits != self.limits
        if changed:
            self._apply_limits(limits)
        L = self.limits[0]

        n_trail = int(TRAIL_S / lab.dt)
        for c, trail, dot, tl in zip(lab.controllers, self.trails, self.dots, self.tilt_lines):
            if not c.enabled or not c.xy:
                for artist in (trail, dot, tl):
                    artist.set_data([], [])
                continue
            xy = np.array(c.xy[-n_trail:])
            trail.set_data(xy[:, 0], xy[:, 1])
            dot.set_data([xy[-1, 0]], [xy[-1, 1]])
            bz = c.core.dyn.R[0, :2, 2]                    # куда наклонён дрон (горизонтальная часть тяги)
            tl.set_data([xy[-1, 0], xy[-1, 0] + 2.5 * L * bz[0]], [xy[-1, 1], xy[-1, 1] + 2.5 * L * bz[1]])

        # стрелка ветра: от наветренного края к центру, длина ∝ скорости
        w = lab.wind
        up = np.array([np.sin(np.radians(w.from_deg)), np.cos(np.radians(w.from_deg))])   # «откуда дует»
        if w.speed > 0.05:
            start = 0.78 * L * up
            self.wind_arrow.xy = tuple(start - 0.25 * L * min(w.speed, 8.0) / 4.0 * up)
            self.wind_arrow.xyann = tuple(start)
            self.wind_arrow.set_visible(True)
            self.wind_label.set_position(tuple(0.62 * L * up + np.array([0, -0.08 * L])))
            self.wind_label.set_text(f"{NAMES_RU[compass_name(w.from_deg)]}, {w.speed:.1f} м/с")
        else:
            self.wind_arrow.set_visible(False)
            self.wind_label.set_text("штиль")
            self.wind_label.set_position((0, 0.6 * L))

        for c, line in zip(lab.controllers, self.err_lines):
            if c.enabled and c.t:
                line.set_data(c.t, 100 * np.linalg.norm(np.array(c.err), axis=1))
            else:
                line.set_data([], [])
        if lab.wind_log:
            tw = np.array([x[0] for x in lab.wind_log])
            ww = np.array([x[1] for x in lab.wind_log])
            self.wind_h.set_data(tw, np.linalg.norm(ww[:, :2], axis=1))
            self.wind_z.set_data(tw, ww[:, 2])

        lines = [f"t = {lab.t:5.1f} с  ветер {np.linalg.norm(w.base()[:2]):.1f} м/с ≈ "
                 f"{100 * self._wind_force_g(w):.0f}% веса",
                 f"{'':>8s}  сейчас  ср.5с  наклон  падений"]
        for c in lab.controllers:
            if not c.enabled:
                continue
            if c.crash_timer > 0:
                lines.append(f"{c.name:>8s}  ПАДЕНИЕ! (всего {c.crashes})")
                continue
            e = np.linalg.norm(np.array(c.err[-int(5 / lab.dt):]), axis=1) if c.err else np.array([np.nan])
            lines.append(f"{c.name:>8s}  {100 * e[-1]:4.1f}см {100 * e.mean():5.1f}см  "
                         f"{c.tilt[-1] if c.tilt else 0:5.1f}°  {c.crashes:4d}")
        self.status.set_text("\n".join(lines))
        return changed

    @staticmethod
    def _wind_force_g(w: LiveWind) -> float:
        """Равновесная сила сопротивления при неподвижном дроне, в долях веса (оценка)."""
        p = CF2X
        v = float(np.linalg.norm(w.base()[:2]))
        rsum = 4 * p.hover_rotor_speed
        force = (rsum * p.rotor_drag_xy + 0.5 * p.air_density * p.body_cda_xy * v) * v
        return force / p.weight

    # ---- блиттинг: каждый кадр — только области трёх графиков; статус (дорогой текст) — 5 раз в секунду
    def _on_draw(self, _event) -> None:
        from matplotlib.transforms import Bbox
        canvas = self.fig.canvas
        self.status_bbox = Bbox.from_extents(0.575, 0.0, 1.0, 0.165).transformed(self.fig.transFigure)
        self.backgrounds = {ax: canvas.copy_from_bbox(ax.bbox) for ax in (self.ax_map, self.ax_err, self.ax_wind)}
        self.backgrounds["status"] = canvas.copy_from_bbox(self.status_bbox)
        self.background = True
        for artist in self.animated:
            (artist.axes or self.fig).draw_artist(artist)

    def render(self, full: bool) -> None:
        canvas = self.fig.canvas
        if full or self.background is None:
            canvas.draw_idle()
            return
        for ax in (self.ax_map, self.ax_err, self.ax_wind):
            canvas.restore_region(self.backgrounds[ax])
            for artist in self.animated:
                if artist.axes is ax:
                    ax.draw_artist(artist)
            canvas.blit(ax.bbox)
        self._frame = getattr(self, "_frame", 0) + 1
        if self._frame % 5 == 0:
            canvas.restore_region(self.backgrounds["status"])
            self.fig.draw_artist(self.status)
            canvas.blit(self.status_bbox)
        canvas.flush_events()

    def tick(self):
        """Шаг анимации: симуляция догоняет реальное время (× скорость), сколько бы ни длился кадр."""
        now = time.perf_counter()
        elapsed = min(now - self._last_wall, 0.2) if self._last_wall is not None else 0.04
        self._last_wall = now
        if self.paused:
            return
        self._sim_debt += elapsed * self.sim_speed
        steps = int(self._sim_debt / self.lab.dt)
        if steps == 0:
            return
        self._sim_debt -= steps * self.lab.dt
        t0 = time.perf_counter()
        self.lab.advance(steps)
        full = self.update()
        if self.interactive:
            self.render(full)
        self.frame_times.append(time.perf_counter() - t0)


def build_controllers(runs: list[str]) -> list[Controller]:
    base_cfg = EnvConfig()
    gains_path = ROOT / "configs" / "pid_tuned.json"
    gains = PIDGains(**json.loads(gains_path.read_text())) if gains_path.exists() else PIDGains()
    ctrls = [Controller("PID", PIDPolicy(base_cfg, gains), base_cfg, style.color("pid")),
             Controller("PD", PIDPolicy(base_cfg, gains.without_integrators()), base_cfg, style.color("pd"))]
    if runs:
        from quadwind.training import load_model
        for i, run in enumerate(runs):
            model, normalizer, cfg = load_model(Path(run), "best")
            method = cfg["name"].split("__")[0]
            name = PRETTY.get(method, method)
            col = style.color(method) if method in style.METHODS else COLORS[(2 + i) % len(COLORS)]
            ctrls.append(Controller(name, SB3Policy(model, normalizer), EnvConfig.from_dict(cfg.get("env")), col))
        ctrls[1].enabled = False                       # PD по умолчанию выключен, если есть модели
    return ctrls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="папка обученной модели (можно несколько)")
    ap.add_argument("--snapshot", default=None, help="сохранить картинку без окна")
    ap.add_argument("--seconds", type=float, default=12.0, help="сколько секунд симулировать для --snapshot")
    ap.add_argument("--speed", type=float, default=3.0, help="начальная скорость ветра, м/с")
    ap.add_argument("--from-deg", type=float, default=0.0, help="откуда дует, ° (0 = С, 90 = В, 180 = Ю, 270 = З)")
    ap.add_argument("--turb", type=float, default=0.3)
    ap.add_argument("--gust-at", type=float, default=None, help="для --snapshot: момент порыва, с")
    ap.add_argument("--autoclose", type=float, default=None, help="закрыть окно через N секунд (проверка)")
    ap.add_argument("--gif", default=None, help="записать анимацию в GIF без окна (длительность — --seconds)")
    ap.add_argument("--fps", type=int, default=10, help="кадров в секунду для --gif")
    args = ap.parse_args()

    if args.snapshot or args.gif:
        import matplotlib
        matplotlib.use("Agg")
    lab = Lab(build_controllers(args.run))
    lab.wind.speed, lab.wind.from_deg, lab.wind.turb_sigma = args.speed, args.from_deg, args.turb
    lab.set_noise(lab.noise_scale)
    view = LabView(lab, interactive=args.snapshot is None and args.gif is None)
    if args.gif:
        from matplotlib.animation import PillowWriter
        steps_per_frame = max(1, int(round(1.0 / (args.fps * lab.dt))))
        gust_step = int(args.gust_at / lab.dt) if args.gust_at is not None else -1
        writer = PillowWriter(fps=args.fps)
        with writer.saving(view.fig, args.gif, dpi=70):
            for k in range(int(args.seconds / lab.dt)):
                if k == gust_step:
                    lab.wind.gust(3.0, 1.0)
                lab.advance(1)
                if k % steps_per_frame == 0:
                    view.update()
                    writer.grab_frame()
        print(f"-> {args.gif}")
        return
    if args.snapshot:
        steps = int(args.seconds / lab.dt)
        gust_step = int(args.gust_at / lab.dt) if args.gust_at is not None else -1
        for k in range(steps):
            if k == gust_step:
                lab.wind.gust(3.0, 1.0)
            lab.advance(1)
        view.update()
        view.fig.savefig(args.snapshot, dpi=110)
        print(f"-> {args.snapshot}")
        return
    timer = view.fig.canvas.new_timer(interval=15)
    timer.add_callback(view.tick)
    timer.start()
    if args.autoclose:                                   # для проверки: закрыть окно через N секунд
        closer = view.fig.canvas.new_timer(interval=int(1000 * args.autoclose))
        closer.single_shot = True
        closer.add_callback(lambda: view.plt.close(view.fig))
        closer.start()
    view.plt.show()
    if view.frame_times:
        ft = np.array(view.frame_times[5:] or view.frame_times)
        print(f"кадров: {len(view.frame_times)}, время кадра: среднее {1e3 * ft.mean():.1f} мс, "
              f"95% {1e3 * np.quantile(ft, 0.95):.1f} мс; симуляция дошла до t = {lab.t:.1f} с")


if __name__ == "__main__":
    main()
