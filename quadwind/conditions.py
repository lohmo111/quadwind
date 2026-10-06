"""Условия эпизода, пространства условий (train / OOD) и фиксированные наборы для оценки.

Разделение данных в этом проекте — это разделение *условий* и *сидов*:
  * train  — непрерывный сэмплинг из TRAIN_SPACE (поток случайных чисел от сида обучения);
  * val, test_id — фиксированные списки из того же пространства, но с другими сидами;
  * ood_*  — фиксированные списки из пространств, которые агент при обучении не видел;
  * сценарии — структурированные тесты (ступенька ветра, сброс груза, развёртка по ветру…).
Наборы строятся детерминированно по имени: одинаковые для всех методов и при каждом запуске.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .params import CF2X
from .wind import COMPASS, WindEvent, WindParams, direction_to_from_deg, from_deg_to_direction

TRAIN_NAMESPACE = 0x7EA1
SUITE_NAMESPACE = 0x5E7


@dataclass(frozen=True)
class U:
    """Равномерный диапазон. `easy` — диапазон при difficulty = 0 (curriculum), `holes` — вырезанные интервалы."""
    lo: float
    hi: float
    easy: tuple | None = None
    holes: tuple = ()

    def bounds(self, difficulty: float = 1.0) -> tuple[float, float]:
        if self.easy is None:
            return self.lo, self.hi
        d = float(np.clip(difficulty, 0.0, 1.0))
        return self.easy[0] + d * (self.lo - self.easy[0]), self.easy[1] + d * (self.hi - self.easy[1])

    def sample(self, rng: np.random.Generator, difficulty: float = 1.0) -> float:
        lo, hi = self.bounds(difficulty)
        for _ in range(1000):
            x = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
            if not any(a <= x <= b for a, b in self.holes):
                return x
        raise RuntimeError(f"cannot sample outside holes {self.holes} in [{lo}, {hi}]")


@dataclass(frozen=True)
class IntU:
    lo: int
    hi: int
    easy: tuple | None = None

    def bounds(self, difficulty: float = 1.0) -> tuple[int, int]:
        if self.easy is None:
            return self.lo, self.hi
        d = float(np.clip(difficulty, 0.0, 1.0))
        return (int(round(self.easy[0] + d * (self.lo - self.easy[0]))),
                int(round(self.easy[1] + d * (self.hi - self.easy[1]))))

    def sample(self, rng: np.random.Generator, difficulty: float = 1.0) -> int:
        lo, hi = self.bounds(difficulty)
        return int(rng.integers(lo, hi + 1))


def fixed(x) -> U:
    return U(x, x)


@dataclass(frozen=True)
class EventSpec:
    """Сценарное событие посреди эпизода.

    kind: 'mass' (value = новая m/m_ном), 'motor' (value = новая эффективность случайного мотора),
          'push' (value = сила в долях m_ном·g, горизонтально), 'wind_step' (value = прибавка ветра, м/с),
          'wind_pulse' (value = пик порыва 1−cos, м/с).
    """
    kind: str
    time: U
    value: U
    dur: U = U(0.0, 0.0)


@dataclass
class EpisodeCondition:
    seed: tuple
    mass_ratio: float = 1.0
    motor_eff: tuple = (1.0, 1.0, 1.0, 1.0)
    motor_tau: float = 0.02
    com_offset: tuple = (0.0, 0.0)
    act_delay: int = 0
    obs_delay: int = 0
    noise_scale: float = 1.0
    wind: WindParams = field(default_factory=WindParams)
    init_radius: float = 0.0
    init_vel_std: float = 0.0
    init_att_std: float = 0.0
    init_rate_std: float = 0.0
    init_yaw: tuple = (0.0, 0.0)
    mass_event: tuple | None = None     # (время, новая m/m_ном)
    motor_event: tuple | None = None    # (время, индекс мотора, новая эффективность)
    push_event: tuple | None = None     # (время, длительность, сила (3,) Н)

    def to_record(self) -> dict:
        """Плоский словарь для CSV (нужен для разрезов результатов по условиям)."""
        w = self.wind
        rec = {
            "seed": "-".join(map(str, self.seed)), "mass_ratio": self.mass_ratio,
            "min_motor_eff": min(self.motor_eff), "motor_tau": self.motor_tau,
            "com_offset": float(np.hypot(*self.com_offset)), "act_delay": self.act_delay,
            "obs_delay": self.obs_delay, "noise_scale": self.noise_scale,
            "wind_speed": w.speed, "wind_dir": w.direction, "wind_from_deg": direction_to_from_deg(w.direction),
            "wind_vertical": w.vertical,
            "wind_jump_rate": w.jump_rate, "slow_sigma": w.slow_sigma, "turb_type": w.turb_type,
            "turb_sigma": w.turb_sigma, "turb_tau": w.turb_tau, "gust_rate": w.gust_rate,
            "init_radius": self.init_radius,
            "mass_event_t": self.mass_event[0] if self.mass_event else np.nan,
            "motor_event_t": self.motor_event[0] if self.motor_event else np.nan,
            "push_event_t": self.push_event[0] if self.push_event else np.nan,
        }
        wind_events = [ev.time for ev in w.events]
        rec["wind_event_t"] = min(wind_events) if wind_events else np.nan
        return rec

    def event_time(self) -> float:
        """Время первого сценарного события (NaN, если событий нет)."""
        times = [ev.time for ev in self.wind.events]
        for e in (self.mass_event, self.motor_event, self.push_event):
            if e is not None:
                times.append(e[0])
        return min(times) if times else np.nan


@dataclass(frozen=True)
class ConditionSpace:
    mass_ratio: U = U(0.8, 1.25, easy=(1.0, 1.0))
    motor_tau: U = U(0.015, 0.04, easy=(0.02, 0.02))
    motor_jitter: U = U(0.0, 0.05, easy=(0.0, 0.0))     # каждый мотор: η = 1 − U(0, jitter)
    weak_motor_prob: U = U(0.25, 0.25, easy=(0.0, 0.0))
    weak_motor_eff: U = U(0.8, 0.95)
    com_offset: U = U(0.0, 0.002, easy=(0.0, 0.0))      # м, направление случайное
    act_delay: IntU = IntU(0, 1, easy=(0, 0))           # шагов управления (20 мс); 2–3 шага — только в OOD
    obs_delay: IntU = IntU(0, 1, easy=(0, 0))
    noise_scale: U = U(0.5, 1.5, easy=(0.5, 0.5))
    wind_speed: U = U(0.0, 3.5, easy=(0.0, 0.0))        # м/с
    wind_direction: U = U(0.0, 2 * np.pi)              # куда дует, рад; сектора по компасу — compass_sectors()
    wind_vertical: U = U(-0.3, 0.3, easy=(0.0, 0.0))
    jump_rate: U = U(0.07, 0.07, easy=(0.0, 0.0))       # ≈ 1 смена режима за эпизод 15 с
    slow_sigma: U = U(0.0, 0.5, easy=(0.0, 0.0))
    slow_tau: U = U(3.0, 10.0)
    turb_type: str = "ou"
    turb_sigma: U = U(0.0, 0.6, easy=(0.0, 0.0))
    turb_tau: U = U(0.2, 1.0)
    turb_vert_ratio: float = 0.5
    gust_rate: U = U(0.0, 0.15, easy=(0.0, 0.0))
    gust_amp: tuple = (1.0, 2.5)
    gust_dur: tuple = (0.5, 2.0)
    init_radius: U = U(0.0, 0.5, easy=(0.0, 0.2))
    init_vel_std: U = U(0.2, 0.2, easy=(0.1, 0.1))
    init_att_std: U = U(0.1, 0.1, easy=(0.05, 0.05))
    init_rate_std: U = U(0.3, 0.3, easy=(0.1, 0.1))
    init_yaw: tuple = (-0.5, 0.5)
    events: tuple = ()

    def sample(self, rng: np.random.Generator, difficulty: float = 1.0, seed: tuple | None = None) -> EpisodeCondition:
        d = difficulty
        eff = 1.0 - rng.uniform(0.0, 1.0, 4) * self.motor_jitter.sample(rng, d)
        if rng.uniform() < self.weak_motor_prob.sample(rng, d):
            eff[rng.integers(4)] = self.weak_motor_eff.sample(rng, d)
        com_r, com_phi = self.com_offset.sample(rng, d), rng.uniform(0.0, 2 * np.pi)

        wind_events, mass_event, motor_event, push_event = [], None, None, None
        for ev in self.events:
            t, v = ev.time.sample(rng), ev.value.sample(rng)
            phi = rng.uniform(0.0, 2 * np.pi)
            horiz = (np.cos(phi), np.sin(phi), 0.0)
            if ev.kind == "mass":
                mass_event = (t, v)
            elif ev.kind == "motor":
                motor_event = (t, int(rng.integers(4)), v)
            elif ev.kind == "push":
                push_event = (t, ev.dur.sample(rng), tuple(v * CF2X.weight * np.array(horiz)))
            elif ev.kind == "wind_step":
                wind_events.append(WindEvent("step", t, tuple(v * np.array(horiz))))
            elif ev.kind == "wind_pulse":
                wind_events.append(WindEvent("pulse", t, tuple(v * np.array(horiz)), ev.dur.sample(rng)))
            else:
                raise ValueError(f"unknown event kind {ev.kind!r}")

        wind = WindParams(
            speed=self.wind_speed.sample(rng, d), direction=self.wind_direction.sample(rng, d),
            vertical=self.wind_vertical.sample(rng, d), jump_rate=self.jump_rate.sample(rng, d),
            jump_speed=self.wind_speed.bounds(d), jump_vertical=self.wind_vertical.bounds(d),
            slow_sigma=self.slow_sigma.sample(rng, d), slow_tau=self.slow_tau.sample(rng, d),
            turb_type=self.turb_type, turb_sigma=self.turb_sigma.sample(rng, d),
            turb_tau=self.turb_tau.sample(rng, d), turb_vert_ratio=self.turb_vert_ratio,
            gust_rate=self.gust_rate.sample(rng, d), gust_amp=self.gust_amp, gust_dur=self.gust_dur,
            events=tuple(wind_events),
        )
        if seed is None:
            seed = (int(rng.integers(2**32)), int(rng.integers(2**32)))
        return EpisodeCondition(
            seed=tuple(seed), mass_ratio=self.mass_ratio.sample(rng, d), motor_eff=tuple(eff),
            motor_tau=self.motor_tau.sample(rng, d), com_offset=(com_r * np.cos(com_phi), com_r * np.sin(com_phi)),
            act_delay=self.act_delay.sample(rng, d), obs_delay=self.obs_delay.sample(rng, d),
            noise_scale=self.noise_scale.sample(rng, d), wind=wind,
            init_radius=self.init_radius.sample(rng, d), init_vel_std=self.init_vel_std.sample(rng, d),
            init_att_std=self.init_att_std.sample(rng, d), init_rate_std=self.init_rate_std.sample(rng, d),
            init_yaw=self.init_yaw, mass_event=mass_event, motor_event=motor_event, push_event=push_event,
        )


def compass_sectors(*sectors: tuple[float, float]) -> U:
    """Направления «откуда дует» из секторов компаса → спецификация ConditionSpace.wind_direction.

    Сектор — (от, до) в градусах по часовой стрелке, можно через север: (315, 45).
    Несколько секторов: compass_sectors((240, 300), (60, 120)) — ветры с запада и востока.
    Промежутки между секторами становятся «дырками» равномерного распределения.
    """
    intervals = []
    for lo, hi in sectors:
        hi = hi + 360.0 if hi < lo else hi
        intervals.append((270.0 - hi, 270.0 - lo))             # «куда дует», градусы
    base = min(a for a, _ in intervals)
    shifted = sorted((a - 360.0 * np.floor((a - base) / 360.0), b - 360.0 * np.floor((a - base) / 360.0))
                     for a, b in intervals)
    holes = tuple((np.radians(b1), np.radians(a2)) for (_, b1), (a2, _) in zip(shifted[:-1], shifted[1:]) if a2 > b1)
    return U(np.radians(shifted[0][0]), np.radians(max(b for _, b in shifted)), holes=holes)


TRAIN_SPACE = ConditionSpace()

# Спокойные условия: без ветра и скрытых изменений, номинальный шум, задержка действия 1 шаг.
# Старт точно в цели — чтобы отклик на событие не смешивался с переходным процессом.
NOMINAL_SPACE = ConditionSpace(
    mass_ratio=fixed(1.0), motor_tau=fixed(0.02), motor_jitter=fixed(0.0), weak_motor_prob=fixed(0.0),
    com_offset=fixed(0.0), act_delay=IntU(1, 1), obs_delay=IntU(0, 0), noise_scale=fixed(1.0),
    wind_speed=fixed(0.0), wind_vertical=fixed(0.0), jump_rate=fixed(0.0), slow_sigma=fixed(0.0),
    turb_sigma=fixed(0.0), gust_rate=fixed(0.0), init_radius=fixed(0.0), init_vel_std=fixed(0.0),
    init_att_std=fixed(0.0), init_rate_std=fixed(0.0), init_yaw=(0.0, 0.0),
)

# Отложенные условия (OOD): за пределами TRAIN_SPACE по одному или нескольким факторам.
OOD_SPACES = {
    "ood_strong_wind": replace(TRAIN_SPACE, wind_speed=U(4.0, 5.0)),
    "ood_heavy": replace(TRAIN_SPACE, mass_ratio=U(1.3, 1.4)),
    "ood_light": replace(TRAIN_SPACE, mass_ratio=U(0.65, 0.75)),
    "ood_weak_motor": replace(TRAIN_SPACE, weak_motor_prob=fixed(1.0), weak_motor_eff=U(0.65, 0.75)),
    "ood_slow_motor": replace(TRAIN_SPACE, motor_tau=U(0.05, 0.07)),
    "ood_dryden": replace(TRAIN_SPACE, turb_type="dryden", turb_sigma=U(0.3, 0.6)),
    "ood_noisy": replace(TRAIN_SPACE, noise_scale=U(2.0, 3.0)),
    "ood_delay": replace(TRAIN_SPACE, act_delay=IntU(2, 3)),
    # «угол» распределения: все факторы одновременно на краю train-диапазона или за ним
    "ood_combined": replace(TRAIN_SPACE, wind_speed=U(3.5, 4.5), mass_ratio=U(1.25, 1.35),
                            noise_scale=U(1.5, 2.0), act_delay=IntU(1, 2)),
}

# Сценарии: одно контролируемое событие в t = 5 с на фоне спокойных условий.
SCENARIO_SPACES = {
    "calm": replace(NOMINAL_SPACE, init_radius=U(0.2, 0.5), init_yaw=(-0.5, 0.5)),
    "step_gust": replace(NOMINAL_SPACE, events=(EventSpec("wind_step", fixed(5.0), fixed(2.5)),)),
    "gust_pulse": replace(NOMINAL_SPACE, events=(EventSpec("wind_pulse", fixed(5.0), fixed(3.0), fixed(1.0)),)),
    "mass_pickup": replace(NOMINAL_SPACE, events=(EventSpec("mass", fixed(5.0), fixed(1.2)),)),
    "mass_drop": replace(NOMINAL_SPACE, mass_ratio=fixed(1.2), events=(EventSpec("mass", fixed(5.0), fixed(1.0)),)),
    "push": replace(NOMINAL_SPACE, events=(EventSpec("push", fixed(5.0), fixed(0.3), fixed(0.3)),)),
    "motor_degrade": replace(NOMINAL_SPACE, events=(EventSpec("motor", fixed(5.0), fixed(0.8)),)),
}

# probe — отдельный набор для анализа памяти (обучение зондов), чтобы не трогать test до заморозки
# safety — 1000 эпизодов для оценки доли падений (на 64 эпизодах падения 1–5% не видны)
SUITE_SIZES = {"val": 256, "test_id": 200, "probe": 256, "safety": 1000, **{k: 100 for k in OOD_SPACES},
               **{k: 32 for k in SCENARIO_SPACES}}
WIND_SWEEP_SPEEDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
WIND_SWEEP_DIRS = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
WIND_COMPASS_SPEED = 3.0       # wind_compass: 8 направлений компаса (С, СВ, …, СЗ) × 3 м/с


def suite_names() -> list[str]:
    return list(SUITE_SIZES) + ["wind_sweep", "wind_compass"]


def _suite_rng(name: str) -> tuple[np.random.Generator, int]:
    key = zlib.crc32(name.encode())   # стабильный хэш (встроенный hash() меняется между процессами)
    return np.random.default_rng([SUITE_NAMESPACE, key]), key


def make_suite(name: str, n: int | None = None, space: ConditionSpace | None = None) -> list[EpisodeCondition]:
    """Детерминированный список условий набора `name`.

    `space` заменяет TRAIN_SPACE для наборов val / test_id / probe (варианты среды в экспериментах):
    сиды те же, распределение — варианта.
    """
    rng, key = _suite_rng(name)
    if name == "wind_sweep":
        conds = []
        for i, (speed, direction) in enumerate((s, d) for s in WIND_SWEEP_SPEEDS for d in WIND_SWEEP_DIRS):
            c = NOMINAL_SPACE.sample(rng, seed=(SUITE_NAMESPACE, key, i))
            c.wind = replace(c.wind, speed=speed, direction=direction)
            conds.append(c)
        return conds[:n] if n else conds
    if name == "wind_compass":
        conds = []
        for i, from_deg in enumerate(COMPASS.values()):
            c = NOMINAL_SPACE.sample(rng, seed=(SUITE_NAMESPACE, key, i))
            c.wind = replace(c.wind, speed=WIND_COMPASS_SPEED, direction=from_deg_to_direction(from_deg))
            conds.append(c)
        return conds[:n] if n else conds
    if name in ("val", "test_id", "probe", "safety"):
        space = space or TRAIN_SPACE
    elif name in OOD_SPACES:
        space = OOD_SPACES[name]
    elif name in SCENARIO_SPACES:
        space = SCENARIO_SPACES[name]
    else:
        raise KeyError(f"unknown suite {name!r}; known: {suite_names()}")
    size = n if n is not None else SUITE_SIZES[name]
    return [space.sample(rng, difficulty=1.0, seed=(SUITE_NAMESPACE, key, i)) for i in range(size)]


def condition_to_dict(c: EpisodeCondition) -> dict:
    return asdict(c)


def override_space(space: ConditionSpace, overrides: dict | None) -> ConditionSpace:
    """Переопределения пространства условий из конфига (YAML).

    Значение для диапазона: [lo, hi] (curriculum-диапазон `easy` сохраняется) или
    {lo, hi, easy: [a, b]}; для строк и кортежей — само значение.
    Пример: {jump_rate: [0, 0], slow_tau: [10, 20], noise_scale: [1.0, 2.5]}.
    """
    changes = {}
    for key, value in (overrides or {}).items():
        current = getattr(space, key)                    # AttributeError на неизвестный параметр — это хорошо
        if isinstance(current, (U, IntU)):
            cast = float if isinstance(current, U) else int
            if isinstance(value, dict):
                lo, hi = cast(value["lo"]), cast(value["hi"])
                easy = tuple(cast(x) for x in value["easy"]) if value.get("easy") is not None else current.easy
            elif isinstance(value, (list, tuple)):
                lo, hi, easy = cast(value[0]), cast(value[1]), current.easy
            else:
                lo = hi = cast(value)
                easy = current.easy
            changes[key] = U(lo, hi, easy=easy, holes=current.holes) if cast is float else IntU(lo, hi, easy=easy)
        elif isinstance(current, tuple):
            changes[key] = tuple(value)
        else:
            changes[key] = value
    return replace(space, **changes)
