"""Единый стиль графиков: цвет закреплён за методом (одинаковый во всех рисунках и в лаборатории).

Палитра — проверенная категориальная (светлая тема): первые четыре цвета различимы попарно
и при нарушениях цветового зрения (валидатор dataviz, все пары). Поэтому на одном графике
с линиями — не больше четырёх главных методов. Остальные показываются в таблицах или
с дополнительным кодированием (штрих, маркер).
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
NEUTRAL = "#8a8985"
BAND = "#eeede8"          # фон «зона обучения» и т. п.

# метод → (подпись, цвет, стиль линии, маркер)
METHODS = {
    "rppo":             ("RPPO", "#2a78d6", "-", "o"),
    "ppo_mlp":          ("MLP", "#eb6834", "-", "s"),
    "ppo_stack16":      ("MLP-k16", "#1baf7a", "-", "^"),
    "pid":              ("PID", "#4a3aa7", "-", "D"),
    "ppo_mlp_integral": ("MLP+∫", "#eda100", "--", "v"),
    "ppo_oracle":       ("Оракул", "#008300", "--", "*"),
    "pd":               ("PD", "#e87ba4", ":", "x"),
    "ppo_stack4":       ("MLP-k4", "#e34948", "--", "<"),
}
PRIMARY = ("rppo", "ppo_mlp", "ppo_stack16", "pid")


def style(method: str) -> tuple[str, str, str, str]:
    return METHODS.get(method, (method, NEUTRAL, "-.", "."))


def label(method: str) -> str:
    return style(method)[0]


def color(method: str) -> str:
    return style(method)[1]


def apply() -> None:
    """Спокойные оси и сетка, нейтральный текст, линии 2 px."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
