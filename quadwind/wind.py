"""Генерация ветра на весь эпизод сразу (скорость воздуха в мировой системе, м/с).

w(t) = базовый (с редкими скачками) + медленный OU + турбулентность (OU или Драйден)
       + случайные порывы «1−cos» + сценарные события (ступенька, импульс, нарастание).

Ряд генерируется при сбросе эпизода из собственного генератора случайных чисел эпизода.
Поэтому реализация ветра одинакова для всех методов при одном сиде и не зависит
от того, упал ли соседний дрон в батче.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

FT = 0.3048


@dataclass
class WindParams:
    speed: float = 0.0                  # базовый горизонтальный ветер, м/с
    direction: float = 0.0              # куда дует, рад от оси x (восток) против часовой; см. from_deg_to_direction
    vertical: float = 0.0               # м/с (+ вверх)
    jump_rate: float = 0.0              # 1/с: скачки базового ветра (смена режима)
    jump_speed: tuple = (0.0, 0.0)      # диапазон новой скорости при скачке
    jump_vertical: tuple = (0.0, 0.0)
    slow_sigma: float = 0.0             # м/с, медленный дрейф (OU)
    slow_tau: float = 5.0               # с
    turb_type: str = "ou"               # "ou" | "dryden"
    turb_sigma: float = 0.0             # м/с, горизонтальная σ турбулентности
    turb_tau: float = 0.5               # с, только для OU
    turb_vert_ratio: float = 0.5        # σ_вертикальная / σ_горизонтальная
    gust_rate: float = 0.0              # 1/с, пуассоновский поток порывов
    gust_amp: tuple = (1.0, 2.0)        # м/с
    gust_dur: tuple = (0.5, 2.0)        # с
    events: tuple = field(default_factory=tuple)  # WindEvent


@dataclass(frozen=True)
class WindEvent:
    """Сценарное событие.

    kind:
      'step'  — постоянная прибавка vec с момента t (ветер включается и остаётся);
      'pulse' — порыв 1−cos с пиком vec длительностью dur;
      'ramp'  — плавное линейное нарастание от 0 до vec за dur секунд, затем vec держится.
    """
    kind: str
    time: float
    vec: tuple            # (3,) м/с: прибавка (step/ramp) или пик порыва (pulse)
    dur: float = 1.0


# ---- направления по компасу ---------------------------------------------------------------------
# Метеорологическая конвенция: направление ветра — ОТКУДА он дует, в градусах от севера по часовой
# стрелке. «Северный ветер» (0°) дует с севера на юг. Мировые оси: x — восток, y — север, z — вверх.
COMPASS = {"С": 0.0, "СВ": 45.0, "В": 90.0, "ЮВ": 135.0, "Ю": 180.0, "ЮЗ": 225.0, "З": 270.0, "СЗ": 315.0}


def wind_from(from_deg: float, speed: float, vertical: float = 0.0) -> np.ndarray:
    """Вектор ветра (м/с, мировые оси) по направлению «откуда дует» и скорости."""
    th = np.radians(from_deg)
    return np.array([-speed * np.sin(th), -speed * np.cos(th), vertical])


def from_deg_to_direction(from_deg: float) -> float:
    """«Откуда дует» (° по компасу) → WindParams.direction («куда дует», рад от оси x против часовой)."""
    return float(np.radians((270.0 - from_deg) % 360.0))


def direction_to_from_deg(direction: float) -> float:
    """WindParams.direction → «откуда дует» (° по компасу, 0…360)."""
    return float((270.0 - np.degrees(direction)) % 360.0)


def ar1(n: int, tau: float, sigma: float, dt: float, rng: np.random.Generator, size: int = 1) -> np.ndarray:
    """Дискретизованный без ошибки OU-процесс, стартующий из стационарного распределения: (n, size)."""
    if sigma <= 0.0:
        return np.zeros((n, size))
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1.0 - a * a)
    eps = rng.standard_normal((n, size))
    x0 = sigma * rng.standard_normal(size)
    out = np.empty((n, size))
    for k in range(size):
        out[:, k] = signal.lfilter([b], [1.0, -a], eps[:, k], zi=[a * x0[k]])[0]
    return out


def dryden_scales(altitude: float, sigma_horizontal: float) -> tuple[float, float, float, float]:
    """Масштабы MIL-F-8785C для малых высот: (L_u, L_w) в м; σ_u = σ_v = sigma_horizontal; σ_w по соотношению.

    Формулы заданы для высоты h ≥ 10 футов; ниже используем 10 футов.
    """
    h = max(altitude / FT, 10.0)
    l_u = h / (0.177 + 0.000823 * h) ** 1.2 * FT
    l_w = h * FT
    sigma_w = sigma_horizontal * (0.177 + 0.000823 * h) ** 0.4
    return l_u, l_w, sigma_horizontal, sigma_w


def _shaped_noise(n: int, num, den, sigma: float, dt: float, burn: int, rng: np.random.Generator) -> np.ndarray:
    """Белый шум через непрерывный фильтр num/den (билинейное преобразование), нормировка на σ."""
    bz, az = signal.bilinear(num, den, fs=1.0 / dt)
    imp_len = int(min(max(4 * burn, 2000), 200_000))
    imp = np.zeros(imp_len)
    imp[0] = 1.0
    gain = sigma / np.sqrt(np.sum(signal.lfilter(bz, az, imp) ** 2))
    eps = rng.standard_normal(n + burn)
    return gain * signal.lfilter(bz, az, eps)[burn:]


def dryden(n: int, dt: float, airspeed: float, altitude: float, sigma_horizontal: float,
           rng: np.random.Generator) -> np.ndarray:
    """Турбулентность Драйдена (u — вдоль среднего ветра, v — поперёк, w — вертикаль): (n, 3).

    Висящий дрон неподвижен относительно земли, поэтому временной масштаб задаётся
    переносом поля средним ветром (гипотеза Тейлора): T = L / V, где V — скорость ветра.
    """
    if sigma_horizontal <= 0.0:
        return np.zeros((n, 3))
    v = max(airspeed, 1.0)
    l_u, l_w, s_u, s_w = dryden_scales(altitude, sigma_horizontal)
    t_u, t_w = l_u / v, l_w / v
    out = np.empty((n, 3))
    out[:, 0] = _shaped_noise(n, [1.0], [t_u, 1.0], s_u, dt, int(5 * t_u / dt), rng)
    for k, (t, s) in enumerate([(t_u, s_u), (t_w, s_w)], start=1):
        num = [np.sqrt(3.0) * t, 1.0]
        den = np.polymul([t, 1.0], [t, 1.0])
        out[:, k] = _shaped_noise(n, num, den, s, dt, int(5 * t / dt), rng)
    return out


def gust_shape(t: np.ndarray, t0: float, dur: float) -> np.ndarray:
    """Профиль 1−cos: 0 вне [t0, t0+dur], пик 1 в середине."""
    phase = (t - t0) / dur
    inside = (phase >= 0.0) & (phase <= 1.0)
    return np.where(inside, 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(phase, 0.0, 1.0))), 0.0)


def _horizontal(speed, direction, vertical) -> np.ndarray:
    return np.array([speed * np.cos(direction), speed * np.sin(direction), vertical])


def generate_wind(wp: WindParams, n: int, dt: float, rng: np.random.Generator,
                  altitude: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (полный ветер, медленная часть) — массивы (n, 3) на моменты t_k = k·dt.

    «Медленная часть» = базовый + скачки + медленный OU + сценарные ступеньки: то, что агент
    в принципе может оценить по истории. Используется в анализе памяти (зонды).
    """
    t = np.arange(n) * dt
    total_time = n * dt

    base = np.tile(_horizontal(wp.speed, wp.direction, wp.vertical), (n, 1))
    if wp.jump_rate > 0.0:
        n_jumps = rng.poisson(wp.jump_rate * total_time)
        for tj in np.sort(rng.uniform(0.0, total_time, n_jumps)):
            new = _horizontal(rng.uniform(*wp.jump_speed), rng.uniform(0.0, 2 * np.pi),
                              rng.uniform(*wp.jump_vertical))
            base[t >= tj] = new

    ratio = np.array([1.0, 1.0, wp.turb_vert_ratio])
    slow = ar1(n, wp.slow_tau, wp.slow_sigma, dt, rng, size=3) * ratio

    if wp.turb_type == "ou":
        turb = ar1(n, wp.turb_tau, wp.turb_sigma, dt, rng, size=3) * ratio
    elif wp.turb_type == "dryden":
        uvw = dryden(n, dt, wp.speed, altitude, wp.turb_sigma, rng)
        c, s = np.cos(wp.direction), np.sin(wp.direction)
        turb = np.stack([c * uvw[:, 0] - s * uvw[:, 1], s * uvw[:, 0] + c * uvw[:, 1], uvw[:, 2]], axis=1)
    else:
        raise ValueError(f"unknown turb_type {wp.turb_type!r}")

    gusts = np.zeros((n, 3))
    if wp.gust_rate > 0.0:
        n_gusts = rng.poisson(wp.gust_rate * total_time)
        for _ in range(n_gusts):
            t0 = rng.uniform(0.0, total_time)
            amp = rng.uniform(*wp.gust_amp)
            dur = rng.uniform(*wp.gust_dur)
            phi = rng.uniform(0.0, 2 * np.pi)
            vec = amp * np.array([np.cos(phi), np.sin(phi), 0.0])
            gusts += gust_shape(t, t0, dur)[:, None] * vec

    events = np.zeros((n, 3))
    for ev in wp.events:
        vec = np.asarray(ev.vec, dtype=float)
        if ev.kind == "step":
            events += (t >= ev.time)[:, None] * vec
        elif ev.kind == "pulse":
            gusts += gust_shape(t, ev.time, ev.dur)[:, None] * vec
        elif ev.kind == "ramp":
            # 0 до ev.time, линейный рост до vec за ev.dur, затем постоянный vec.
            # Попадает в «медленную» часть ветра: агент может оценить его по истории.
            progress = np.clip((t - ev.time) / max(ev.dur, 1e-9), 0.0, 1.0)
            events += progress[:, None] * vec
        else:
            raise ValueError(f"unknown wind event {ev.kind!r}")

    slow_part = base + slow + events
    return slow_part + turb + gusts, slow_part


class LiveWind:
    """Ветер для интерактивного режима: параметры можно менять на лету, турбулентность (OU) — онлайн.

    Используется в scripts/wind_lab.py вместо заранее сгенерированного ряда.
    """

    def __init__(self, dt: float, seed: int = 0):
        self.dt = dt
        self.speed = 0.0            # м/с
        self.from_deg = 0.0         # откуда дует, ° по компасу
        self.vertical = 0.0         # м/с (+ вверх)
        self.turb_sigma = 0.0       # м/с, горизонтальная σ
        self.turb_tau = 0.5         # с
        self.turb_vert_ratio = 0.5
        self.t = 0.0
        self._gusts: list[tuple[float, float, np.ndarray]] = []
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self._turb = np.zeros(3)
        self.t = 0.0
        self._gusts.clear()

    def base(self) -> np.ndarray:
        return wind_from(self.from_deg, self.speed, self.vertical)

    def gust(self, amplitude: float, duration: float = 1.0, from_deg: float | None = None) -> None:
        """Порыв 1−cos с пиком `amplitude` м/с с текущего момента (по умолчанию — по текущему направлению)."""
        direction = self.from_deg if from_deg is None else from_deg
        self._gusts.append((self.t, duration, wind_from(direction, amplitude)))

    def step(self) -> np.ndarray:
        a = np.exp(-self.dt / self.turb_tau)
        ratio = np.array([1.0, 1.0, self.turb_vert_ratio])
        self._turb = a * self._turb + np.sqrt(1.0 - a * a) * self.turb_sigma * ratio * self.rng.standard_normal(3)
        gust = np.zeros(3)
        for t0, dur, vec in self._gusts:
            gust += float(gust_shape(np.asarray(self.t), t0, dur)) * vec
        self._gusts = [g for g in self._gusts if self.t <= g[0] + g[1]]
        self.t += self.dt
        return self.base() + self._turb + gust
