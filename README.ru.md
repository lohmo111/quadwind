# quadwind — удержание позиции квадрокоптера в ветре (Recurrent PPO)

Учебный проект: агент на Recurrent PPO (LSTM) удерживает Crazyflie 2.x в точке при порывистом ветре,
неизвестной массе, ослабленном моторе, шуме датчиков и задержках. Скрытые условия агент должен
оценивать по истории наблюдений.

* `research_rppo_drone_hover.html` — исследование: постановка, обоснования, литература.
* `key_takeaways.md` — главные выводы исследования.
* `PLAN.md` — план работ с контрольными точками и статусом.

## Установка (Windows / Linux / Colab)

```bash
python -m venv .venv                 # Python 3.10+; проверено на 3.13
.venv/Scripts/activate               # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                               # ~95 тестов, ~40 с (первый запуск дольше: компиляция numba)
```

pybullet не нужен: симулятор свой (NumPy + numba), параметры Crazyflie взяты из gym-pybullet-drones.

## Структура

| Путь | Что внутри |
|---|---|
| `quadwind/params.py` | Параметры Crazyflie CF2X |
| `quadwind/kernels.py` | JIT-ядра: физика, моторы, миксер, регулятор угловых скоростей |
| `quadwind/dynamics.py` | Состояние N дронов, описание модели |
| `quadwind/wind.py` | Ветер: базовый + скачки + медленный OU + турбулентность (OU / Драйден) + порывы |
| `quadwind/conditions.py` | Условия эпизода, train-пространство, наборы val / test_id / ood_* / сценарии |
| `quadwind/env.py` | Среда: батчевое ядро, SB3 VecEnv, gymnasium.Env |
| `quadwind/control.py` | Каскадный PID-baseline (через тот же CTBR-интерфейс) |
| `quadwind/policies.py` | Адаптеры политик для оценки (PID, SB3, с передачей LSTM-состояния) |
| `quadwind/evaluation.py`, `metrics.py`, `stats.py` | Прогон, метрики по истинному состоянию, IQM + бутстрап |
| `quadwind/training.py`, `fast_lstm.py` | Сборка обучения, колбэки, ускоренная LSTM-политика |
| `quadwind/memory.py` | Зонды и вмешательства в скрытое состояние LSTM |
| `configs/` | `base.yaml` + конфиги методов (M2–M6 из PLAN.md), `pid_tuned.json` |
| `quadwind/style.py` | Цвета методов (проверенная палитра) и стиль графиков |
| `scripts/` | `train.py`, `run_series.py`, `evaluate.py`, `compare_variants.py`, `make_figures.py`, `tune_pid.py`, `analyze_memory.py`, `wind_lab.py` |

## Запуск

```bash
# обучение (один метод, один сид); результаты в runs/<exp>/<method>/seed_<k>/
python scripts/train.py --config configs/rppo.yaml --seed 0 --exp main --threads 2
tensorboard --logdir runs/main

# серия: конфиги × сиды, 4 процесса параллельно (готовые запуски пропускаются)
python scripts/run_series.py --exp main --seeds 0 1 2 3 4 --parallel 4 --threads 3 \
    --configs configs/ppo_mlp.yaml configs/ppo_stack16.yaml configs/ppo_oracle.yaml configs/rppo.yaml

# анализ памяти RPPO: зонды + обнуление/подмена скрытого состояния
python scripts/analyze_memory.py --run runs/main/rppo/seed_0

# варианты среды (оверлеи configs/e7/*.yaml, configs/hp/*.yaml) и их сравнение
python scripts/run_series.py --exp e7 --seeds 0 --configs configs/rppo.yaml configs/ppo_mlp.yaml     --overlays configs/e7/v1_precise.yaml configs/e7/v2_persistent.yaml
python scripts/compare_variants.py --runs "runs/e7/*/seed_0" --out results/e7

# рисунки для отчёта (по CSV из evaluate.py / compare_variants.py и по моделям)
python scripts/make_figures.py --csv results/e7/episodes.csv --runs runs/main/rppo/seed_0 --out results/figures

# подбор PID на val
python scripts/tune_pid.py --out configs/pid_tuned.json

# оценка: модели + baseline на наборах (CSV: один эпизод — одна строка)
python scripts/evaluate.py --runs "runs/main/*/seed_*" --baselines pid pd --pid-gains configs/pid_tuned.json \
    --suites test_id ood scenarios --out results/main.csv
```

## Ветровая лаборатория (интерактивно, с ползунками)

```bash
python scripts/wind_lab.py                                    # PID и PD
python scripts/wind_lab.py --run runs/pilot2/rppo/seed_0 --run runs/pilot2/ppo_mlp/seed_0
python scripts/wind_lab.py --snapshot lab.png --seconds 12 --from-deg 270 --speed 4   # картинка без окна
python scripts/wind_lab.py --gif lab.gif --seconds 15 --gust-at 6 --run runs/pilot2/rppo/seed_0         # анимация
```

* **Роза ветров** — кнопки С, СВ, В, ЮВ, Ю, ЮЗ, З, СЗ. Направление — **откуда дует** (северный ветер дует с севера на юг), как в метеосводках. На карте x — восток, y — север.
* **Ползунки:** сила ветра, точное направление (°), турбулентность, вертикальный ветер, сид, масса, мотор №1 (ослабить), шум датчиков, задержка, скорость симуляции.
* **Кнопки:** «Порыв!» (+3 м/с на 1 с по текущему направлению), «Толчок» (0,3 веса поперёк ветра), «Сброс», «Пауза».
* Все отмеченные регуляторы летят **в одном и том же ветре с одним и тем же шумом**, поэтому сравнение честное. Линия у дрона показывает, куда он наклонён: при ветре — навстречу ветру.

В обучающей среде направление задаётся так:
`ConditionSpace(wind_direction=compass_sectors((315, 45)))` — только ветры с северной стороны (от СЗ до СВ).
Для разовых векторов — `quadwind.wind.wind_from(from_deg, speed)`.

## Скорость (Windows, 12 потоков CPU)

| Что | Скорость |
|---|---|
| Среда, N = 16 / 64 / 256 | ~40 тыс. / ~90 тыс. / ~130 тыс. шагов/с |
| Обучение PPO-MLP | ~6 тыс. шагов/с (10 млн шагов ≈ 30 мин) |
| Обучение RecurrentPPO (LSTM 64, n_steps 512) | ~2 тыс. шагов/с (10 млн шагов ≈ 1,5 ч) |
| Оценка 64 эпизодов по 15 с | ~1–2 с |

Несколько запусков можно вести параллельно (`--threads 2` на процесс).
