TL;DR

Завдання: визначити амплуа (позицію) гравця за атрибутами FIFA.

Фічсети:
A-48 — фіксований компактний набір (48 ознак);
B-42 — авто-відбір 42 найінформативніших фіч (MI + RF з CV).

Сценарії даних: no_leak (чисті фічі) та with_leak (додаємо club_position / nation_position з RAW для аналізу; не для прод).

Моделі: плоскі (LR, RF, HGB, XGB*), ієрархічні (group→position), ансамблі (soft-vote).

Ключові метрики (single-label):
acc_any (вгадали будь-яку з реальних позицій гравця), top2_any, macro_f1, balanced_acc, group_acc, avg_cost_any.

Мультилейбл (OVR): оптимізація порогів + вартісна функція з штрафами за «хибні» та «чужі за групою» позиції.

Артефакти: усі результати зберігаються в out/tables/…, модель — у models/.

* XGB використовується за наявності бібліотеки; інакше — фолбек на HGB.

Ноутбуки (відкрити у Colab)

01 — EDA та формування таргетів:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/01_eda_and_labeling.ipynb

02 — Аналітичний блок:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/02_analytical_block.ipynb

03 — Візуалізації:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/03_visualization_block.ipynb

04 — Аналітичне мислення:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/04_analytical_thinking_block.ipynb

05 — Бізнес-аналітика:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/05_business_analytics_block.ipynb

06 — Data Science (інженерія ознак + авто-відбір):
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/06_data_science_block.ipynb

07 — Моделювання (Task20), експорт і синхронізація:
https://colab.research.google.com/github/rvkushnir/project_fifa_players/blob/main/notebooks/07_data_science__models_block.ipynb

Структура репозиторію

### Структура репозиторію

project_fifa_players/
├─ data/
│  ├─ raw/                # сирі дані (не в репо; див. README.md та .gitkeep)
│  ├─ interim/            # проміжні (ігнорується Git)
│  └─ processed/          # оброблені (ігнорується Git)
├─ out/
│  ├─ tables/             # CSV-артефакти (потрапляють у Git)
│  └─ figures/            # графіки (PNG)
├─ models/
│  ├─ best_model.pkl      # запакована прод-модель (whitelist у .gitignore)
│  └─ metadata.json       # метадані експорту
├─ notebooks/
│  └─ ...                 # ноутбуки 01–07
├─ scripts/
│  └─ export_best_model.py
├─ app.py                 # (опціонально) Streamlit-додаток "Дізнай своє амплуа"
├─ config.yaml            # конфіг для сервісу/скриптів
├─ requirements.txt
└─ README.md              # цей файл

Дані

Джерело: офіційні/напівофіційні витяги атрибутів FIFA (прикладом — sofifa_id, переліки player_positions, технічні атрибути).

Сирі дані (data/raw/) не комітяться; у репозиторії — лише заглушки (.gitkeep/README.md).

В ноутбуку 06 формується чистий train-набір out/tables/Task18_features_train.csv:
нормалізація позицій до 15 канонічних, перевірка достатності ключових індексів, фільтрація жорстких фінансових аномалій тощо.

Інженерія ознак і відбір фіч (ноутбук 06)

Фічсет B (детальний пул): усі “технічні” префікси
attacking_*, skill_*, movement_*, mentality_*, power_*, defending_*, goalkeeping_*

базовий профіль (overall, potential, age, height_cm, weight_kg, weak_foot, skill_moves, international_reputation, work_rate, preferred_foot, body_type, work_att, work_def).

Автовідбір (Task19): комбінуємо Mutual Information (нелінійні зв’язки) і RandomForest importance (усереднено по 5-фолд CV).
Категоріальні фічі one-hot’яться (щільна матриця), важливості агрегуються назад на “сиру” фічу.
Після зведеного рейтингу дропаємо надлишково корельовані числові (|r| ≥ 0.95).
На виході: B-42 (топ-42).

Фічсет A-48 — компактний фіксований (48 фіч): базовий профіль + індекси (pace_idx, dribble_idx, playmake_idx, attack_idx, defend_idx, phys_idx, gk_idx) + помірна частка детальних навичок. Формується прямо в моделюванні (ноутбук 07), окремий файл списку не зберігається.

Артефакти 06/19:

out/tables/Task18_features_train.csv — train-матриця;

out/tables/Task18_featset_B_list.csv — детальний пул;

out/tables/Task19_selected_features_B.csv — B-42;

маніфест _manifest_tasks_17_19.csv та супутні EDA-CSV.

Моделювання (Task20, ноутбук 07)
Сценарії даних

no_leak — тільки обрані фічі.

with_leak — додаємо з RAW: club_position / nation_position.
Корисно для порівняння “upper-bound”, не використовуємо в прод.

Підходи

Плоскі моделі:
Logistic Regression (LBFGS), RandomForest, HistGradientBoosting, XGBoost* (якщо доступний; інакше — HGB).

Ієрархічні: 2-кроковий підхід “group→position”
(спершу група {GK, DEF, MID, FWD}, далі — позиція всередині групи; агрегуємо ймовірності).

Ансамблі: soft-vote по ймовірностях кількох flat-моделей (LR+HGB; LR+HGB+XGB).

Крос-валідація: StratifiedKFold(n_splits=5, seed=42).
Калібрування: CalibratedClassifierCV(..., method="isotonic") для дерев/бустінгів.

Метрики (single-label)

acc_any — чи входить найімовірніша передбачена позиція у множину реальних позицій гравця (player_positions).

top2_any — чи входить хоч одна з top-2 передбачених позицій у множину реальних.

macro_f1, balanced_acc — класичні класифікаційні.

group_acc — точність на рівні груп {GK, DEF, MID, FWD}.

avg_cost_any — середня “вартість” помилки:
0 якщо вгадали будь-яку реальну позицію; 1 якщо промах у межах тієї ж групи; 3 — крос-груповий промах.

Ми не використовуємо acc_primary і top2_acc, бо бізнес-логіка не прив’язує “первинну” позицію як більш важливу; важливо вгадати будь-яку валідну позицію для гравця.

Мультилейбл (OVR)

One-Vs-Rest поверх базового класифікатора (LR або HGB) + калібрування (де доречно).

Підбір порогів per-label на валідації (max F1 по сітці t∈{0.2,0.3,0.4,0.5}).

Метрики: subset_acc, micro_f1, macro_f1, hamming_loss, jaccard_samples, acc_any,
та вартісна метрика (штрафи за “зайві”/“пропущені” та “чужі за групою”, GK↔польові — підвищений штраф).

Підсумкові CSV: out/tables/Task20_multilabel_folds.csv, out/tables/Task20_multilabel_results.csv.

Результати та артефакти (single-label)

По всіх моделях/сценаріях:
out/tables/Task20_FULL_runs_detailed.csv,
out/tables/Task20_FULL_results_summary.csv (усереднено по фолдах).

Експорт найкращої моделі:
models/best_model.pkl + models/metadata.json (версія, фічсет, сценарій, метрики, дата).

.gitignore налаштовано з allowlist — ці файли комітяться.

Як відтворити локально

Середовище
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Покласти сирі дані у data/raw/ (див. опис у data/raw/README.md).
Запустити ноутбуки:
06_data_science_block.ipynb — підготовка train-матриці, фічсет B, авто-відбір B-42, маніфест.
07_data_science__models_block.ipynb — навчання Task20 (A-48 і B-42; no_leak/with_leak), мультилейбл, експорт models/….
(Опціонально) Стрімліт-демо - streamlit run app.py

Streamlit: “Дізнай своє амплуа” (опціонально)

Вкладки

«Калькулятор амплуа»: ручне введення ключових скілів → top-1/top-3 позиції + «роль-кохерентність».
«Топ-листи»: пенальтісти, GK проти пенальті, «топ-11» тощо.
«Про модель»: опис фіч/метрик, дата тренування, обмеження.
Артефакти: models/best_model.pkl, config.yaml, кеш через st.cache_data / st.cache_resource.
Безпека демо: без персональних даних; тільки агрегати та синтетичні введення.
Відтворюваність

Всі рандомні процедури фіксуються random_state=42.
Відбір фіч (MI+RF) і моделі оцінюються в 5-фолд CV.
Маніфести задач (_manifest_*.csv) дозволяють швидко перевірити повноту артефактів.

Чому так?
Фокус метрик — acc_any / top2_any і вартісні (avg_cost_any), оскільки бізнес-задача не вимагає вгадувати «первинну» позицію, достатньо будь-якої коректної.
Ієрархія (group→position) додає стабільності та дозволяє контролювати «крос-групові» помилки, які й дорожчі у вартісній функції.
MI + RF CV виявляють як нелінійні, так і інтерактивні залежності; кореляційний фільтр зменшує мультиколінеарність і дає компактніші моделі.
Підсумкові числа дивіться у out/tables/Task20_FULL_results_summary.csv (single-label) та out/tables/Task20_multilabel_results.csv (multilabel).
