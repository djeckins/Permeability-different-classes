# LAB RAT.AI — аудит главной страницы и спецификация наполнения

Дата: 16 сентября 2026.
Задача: проанализировать первую страницу сайта и предложить наполнение по образцу
playmolecule.ai, включая блок Technology stack.

Полная версия с макетами предлагаемых секций опубликована отдельной страницей:
<https://claude.ai/artifact/4mxoHhfopVymk75Zbb1Nm2>

---

## Короткий вывод

Плотность playmolecule — это **не демо и не бенчмарки, а инвентарь**. На их главной
нет ни одного интерактивного элемента, ни одной таблицы бенчмарков, ни одного
графика и ни одного кейса с названием клиента. Ощущение технической достоверности
создают четыре вещи:

1. названный каталог из 24 инструментов, каждый с одним предложением, содержащим
   название метода;
2. четыре вставляемых примера запросов с настоящими PDB-кодами, SMILES, мишенями
   и единицами;
3. три статьи с DOI;
4. трёхслойная схема стека с однострочным резюме.

Ничего из этого не требует строить продукт. Требуется опубликовать то, что уже
есть внутри сервиса. Страница «How It Works» уже содержит пятистадийный конвейер,
который у playmolecule называется Skills.

Единственный структурный блокер — строка в FAQ: «CAN WE SEE THE METHOD? Under NDA,
yes. Not on this site. Its absence here is deliberate.» Пока она стоит, плотность
playmolecule недостижима. playmolecule решил ровно эту проблему разделением: они
публикуют **имена инструментов и методные существительные**, но скрывают
композицию протокола, и говорят об этом прямо — «The internal composition of the
protocol is maintained by us».

---

## Что на главной сейчас

Снапшот содержит четыре страницы. Главная — 58 КБ разметки, семь секций,
22 изображения (все декоративные), один информативный элемент (canvas с точками).

| Секция | Объём | Содержание | Оценка |
|---|---|---|---|
| 01 The run | 5.8 КБ | H1, CTA, «One run, in numbers», пример IN→/OUT→ | числа не о вас |
| 02 The arithmetic | 12.2 КБ | €110B / €1.4B / ≈€250B, «≈180×», canvas 40 000 точек | три числа неверны |
| *03 — отсутствует* | — | дыра в нумерации `data-screen-label` | артефакт |
| 04 The department | 4.3 КБ | только заголовок «Not one model. A department.» | пустая |
| 05 November 2026 | 3.0 КБ | новые классы опасности EU, запрет 2013 | сильнейшая секция |
| 06 What we sell | 5.0 КБ | три карточки аудиторий | рабочая |
| 07 Close | 4.5 КБ | повтор CTA | дубль |

Шесть наблюдений:

- **Нет предмета.** «8 специалистов» — число без имён. «A department» — заголовок
  без состава: секция состоит из одной строки.
- **Ни одного числа о собственной работе.** Все цифры про рынок и про фарму.
- **Ни одного доказательства.** Три источника в сноске — все про размер рынка.
- **Ни одного выхода в глубину.** С главной ведут шесть ссылок, из них ни одной
  в методы, документацию или каталог.
- **Нечего попробовать.** Единственное действие — заявка и ожидание двух дней.
- **Все визуалы декоративные.**

Что сохранить: копирайт, ритм, секцию «November 2026» и интонацию юридических
оверлеев. Строка «We are *not certified today*, and we would rather say so here
than let a badge imply an audit that has not happened» — лучшая на сайте и главный
актив достоверности.

---

## Три сайта, которые ближе к вам, чем playmolecule

playmolecule — платформа с каталогом и тарифом Pro. Вы — сервис с одним вопросом
на клиента и дозье через три недели. Форма подачи переносится, но структурно
ближе другие.

**Axiom Bio (axi.om) — прямой ответ на блокер с методом.** In-silico
токсикология, высокий тач, без самообслуживания. Навигация из **двух** пунктов:
Translate и **Methods**. У них метод тоже закрыт, и решили они это страницей
Methods, которая называет базы, движки, QC-шаги и шесть биологических ридаутов с
конкретными красителями (Steatosis — Hoechst / LipTOX; ER Stress — BIP/GRP78;
Cholestasis — Phalloidin / Actin), не раскрывая композицию. Плюс датированная
лестница артефактов на главной: 03/2026 Technical Report → 06/2026 Model Card →
08/2026 Technical Summary. Каденция сообщает «программа живая» лучше любой прозы.

Их приём Stop / Advance стоит скопировать буквально — анонимный пример с
*рассуждением*, а не вердиктом. Переложенный на ваш домен:

> **Deprioritised · PPAR-γ agonism risk** — planar hydrophobic surface · docking
> pose seated in the LBD · read-across to three known agonists at Tanimoto > 0.6
>
> **Advanced · same question** — higher 3D character · no LBD pose across the
> ensemble · nearest known agonist at Tanimoto 0.21

**Deep Origin (deeporigin.com/validated-assets) — артефакт, которого не хватает.**
Единственный найденный образец ретроспективной валидации. Структура: Challenge →
What we did → Result, с процитированной базовой линией из литературы (hit rate
индустрии ≤1%, с DOI), фальсифицируемой воронкой (80 млрд отскринено → 183
протестировано → 56 хитов → 14 клеточно активных → лид 570 нМ), метрикой новизны
(ECFP4 Tanimoto 0.22–0.27) и гейтом «shared under CDA on request». Они называют
конкурентов поимённо, ссылаются на сторонний лидерборд и добровольно указывают,
где проигрывают («Tradeoffs remain on CYP-mediated inhibition»). Это убеждает
именно потому, что это можно атаковать.

**Arcaea (arcaea.com) — предостережение.** $78 млн привлечено, Chanel и Givaudan
в инвесторах, главная состоит из манифеста: «A beauty revolution through
biology», «we call it expressive biology». Ноль чисел, ноль статей, ноль имён,
ноль инструментов. В футере «© Copyright 2024». Это структурно ровно то, где
сейчас ваша главная. Манифест сам по себе не покупает ничего: Isomorphic Labs
может позволить себе чистый манифест только потому, что за ним AlphaFold и
Хассабис.

**Конкурент, которого надо знать.** Potion AI продаёт ровно вашей аудитории и уже
отгружает: поиск на естественном языке по 80 000 сырьевых материалов,
реверс-инжиниринг INCI-листа в стартовую формулу с торговыми названиями и
концентрациями, бесплатный тариф $0/месяц, SOC 2 Type II и двенадцать именных
отзывов с должностями — Olaplex, THG Labs, Goodier Cosmetics, Herrco. При этом ни
одной статьи и ни одного бенчмарка: достоверность держится целиком на бесплатном
входе и именах практиков. Это третья стратегия доверия, и в вашей отрасли она
работает.

**Как покупатель на самом деле ищет.** Skinobs — де-факто индекс отрасли:
1 500+ методов, 500+ провайдеров, 8 000 пользователей из 98 стран. Поиск устроен
по осям **claim → biomarker → assay support → бюджет**, и in silico там отдельная
категория преклиники. Отзывы от Dior, Sisley, Clarins, Clariant, Lubrizol — с
должностями. Ингредиентные дома навигируют так же: у Mibelle 18 фильтров по
трендам, а единица доверия — названная молекула, проиндексированная под claim.

Ваша страница организована по оси «биологический вопрос → молекулы». Отрасль ищет
по оси «claim → метод». Страница на семейство claim'ов (барьер, увлажнение, тон,
фотостарение, долговечность волос) с честным разбором, что вычислительный маршрут
по этому claim'у устанавливает, а что нет, найдёт больше читателей, чем любой
объём платформенного копирайта.

---

## Technology stack — предлагаемая секция

Три слоя. Нижний — отличие от фармы.

```
03  THE CO-SCIENTIST
    Turns a claim into a biological question, decides which instruments to run
    and in what order, reads back what came out, and stops the run when any
    specialist objects. It never computes, and it never asserts biology without
    a citation you can open.
    Question framing · Run planning · Evidence assembly · Stop conditions

02  THE INSTRUMENTS
    Twenty-four validated computational applications, grouped by the stage of
    the question they answer. Each is a single job with a defined input and a
    defined output; none of them decides anything.
    Descriptor and ionisation profiling · Implicit-membrane transport ·
    Ensemble docking · All-atom dynamics · Structural alert catalogues ·
    Structure and substructure search

01  THE REGULATORY SUBSTRATE
    What is allowed to exist, per market, on the day you ask. Roughly 30,400
    named cosmetic ingredients, the prohibited and restricted annexes as they
    change, the patent and literature corpora, and characterised natural-product
    space.
    EU · UK · US by default · Annexes II–VI tracked on revision ·
    Prior art by structure
```

Строка-резюме под схемой:

> The co-scientist decides what to run. The instrument layer runs the physics.
> The regulatory layer decides what is allowed to exist at all — and that is the
> layer pharma tooling was never built to carry.

Это самое ценное в структуре. Секция «The arithmetic» сейчас утверждает, что у вас
те же инструменты, что у фармы, только дешевле — то есть делает вас бедной фармой.
Регуляторный слой переворачивает аргумент: у вас есть слой, которого у фармы нет.

---

## Каталог возможностей — 24 позиции по пяти стадиям

Заменяет пустую секцию «The department». Имена рабочие, требуют переназывания под
голос бренда. Пометка «runs today» — проверено запуском.

**01 · Target & mechanism**
- ClaimMap — turns a marketing claim into a named biological endpoint and the proteins that carry it
- TargetTriage — splits the protein set into three lists: target, adjacent, must-never-hit
- EvidenceReader — extracts mechanism claims from the literature with the species each was shown in
- PathwayContext — places the target in its pathway and flags the feedback that undoes the effect

**02 · Search & repurpose**
- AnnexSearch — searches named-ingredient space against the live prohibited and restricted annexes, per market
- NaturalSpace — searches characterised natural-product space for the same pharmacophore
- StructureSearch — Morgan-fingerprint similarity and substructure search across chemical databases
- Repurpose — new purposes for molecules already cleared; cleared molecules nobody tried for this

**03 · Prove**
- PhysChem *(runs today)* — descriptor panel: MW, TPSA, HBD/HBA, rotatable bonds, heavy atoms, formal charge, cLogP
- pKaNet *(runs today)* — graph-neural-network protonation prediction; ionisation class and unionised fraction at the pH of the site
- Permeation *(runs today)* — implicit-membrane transport panel: black lipid membrane, PAMPA, Caco-2, plasma, blood–brain barrier
- BarrierScore *(runs today)* — weighted scoring for passive stratum corneum entry, 0–100, with a verdict per criterion
- Alerts *(runs today)* — PAINS and BRENK structural alert catalogues, reported as flags rather than scored
- EnsembleDock — docking against the pocket across a conformational ensemble, not a single crystal pose
- ComplexMD — all-atom dynamics on the complex; binding stability and residence time
- SafetyPanel — sensitisation, genotoxicity and CYP450 interaction

**04 · Formulate**
- BaseComposer — composes base, emulsifier, preservation and pH from a claim, a texture and a price per kilo
- StabilityMap — behaviour across pH, oxidation and light
- Compatibility — interaction with common actives sharing the phase
- BenchSheet — the formula written as a bench sheet, per market, in the order things go in

**05 · Plan**
- PriorArt — patent search by structure and by claim, never by trade name
- FTO — freedom to operate per market, with the nearest prior art named rather than summarised
- DesignAround — where the space is crowded: a route around it, or a licence route
- AssayDesign — the cheapest in vitro experiment that could still prove it wrong, with the number that would fail it

**Условие публикации:** см. `engine-findings-for-website.md`. Четыре из пяти
«маршрутов применения» в движке сегодня — один и тот же файл, и постоянно
заряженные катионы ошибочно проходят скрин кожи. Каталог в стиле playmolecule
работает ровно потому, что каждая строка проверяема.

---

## Примеры запросов

Самый дешёвый приём с наивысшей отдачей на слово. Требует нуля разработки.

> **Target · mechanism.** Find a permitted molecule that raises aquaporin-3
> expression in keratinocytes without PPAR-γ agonism, cleared for leave-on use in
> the EU, UK and US.

> **Prove.** Rank these 40 INCI-listed botanical extracts by predicted stratum
> corneum passage at pH 5.5, and flag any that carry a permanent charge.

> **Formulate.** Compose a preservative system for a pH 5.5 O/W emulsion that
> avoids phenoxyethanol, is compliant in the EU and Japan, and costs under €4 per
> kilo.

> **Plan.** Our lead sits on Annex III with a 0.5% limit. Find the nearest
> structural analogue that does not, and tell me who owns it.

Регистр — INCI, Annex III, pH 5.5, leave-on, O/W, €/кг, названия белков — словарь,
который невозможно подделать, не работая в отрасли.

---

## Числа, которые надо исправить

Проверены все восемь количественных утверждений главной. Три неверны, и как раз
они стоят крупным кеглем.

| На странице | Вердикт | Что не так |
|---|---|---|
| €110B рынок ЕС, 2025 | верно | Cosmetics Europe, розничные цены. 2024 — €104 млрд |
| €1.4B R&I L'Oréal, 3.1% | верно | FY2025: €44.05 млрд продаж, R&I €1 381 млн, 3.13% |
| ≈€250B R&D фармы | **устарело** | Цифра за 2021 из двухстраничной заметки 2024. $276 млрд по курсу 2021 = **€233 млрд**. Заменить на «≈$300 млрд, 2025» |
| ≈180× | **ошибка категории** | Сравнивает одну компанию с целой отраслью. Замена: «вся европейская косметическая отрасль тратит на R&D за год меньше, чем одна фарм-компания из первой десятки» (€2.35 млрд против $14.7 млрд у J&J) |
| 40 000 разрешённых, CosIng | **число и слово** | В глоссарии **30 418** записей (Решение ЕС 2025/1175). «Разрешённых» неверно принципиально: ЕС работает по негативному списку, CosIng включает запрещённые аннексы. Замена: «~30 400 названных ингредиентов — и всего ~60 разрешённых консервантов и меньше 30 UV-фильтров» |
| ≈10⁵⁰ молекул | **нет источника** | Канон: **10⁶⁰** (Bohacek, McMartin & Guida, 1996) или **~10³³** для липински-совместимых (Polishchuk, Madzhidov & Varnek, 2013) |
| Ноябрь 2026 | верно | Регламент (ЕС) 2023/707, вещества на рынке до 1 мая 2025 — с 1 ноября 2026. Дописать vPvB и vPvM |
| С 2013 не доказать по-старому | нужна оговорка | Дата верна. Но T-655/20 и T-656/20 (Symrise против ECHA, 22.11.2023) разрешили ECHA требовать тесты на животных по REACH. Безопаснее: «нельзя *получить одобрение* по-старому» |

Что поставить взамен:

| Число | Утверждение |
|---|---|
| **<30** | UV-фильтров одобрено на весь ЕС; последний новый (S87) вступил в силу 3 декабря 2020 — шесть лет без единого нового |
| **14** | Итоговых заключений SCCS за весь 2024 год — вся экспертная пропускная способность ЕС для отрасли в €110 млрд |
| **30 418** | Записей в глоссарии, +348 за одно обновление; этикетка может перестать соответствовать без изменений в формуле |
| **1 766 / 0** | Запрещённых позиций в Annex II; Omnibus VIII применяется с 1 мая 2026 **с нулевым периодом распродажи** |
| **~7 000** | Веществ может быть ограничено к 2030 (Restrictions Roadmap), и ограничиваются они **группами** |
| **12-я ред.** | SCCS Notes of Guidance SCCS/1647/22 от 15.05.2023 формально вводит NAM, NGRA, TTC, read-across и **in silico / QSAR** |
| **€2.35B** | Годовые вложения в R&I всей европейской косметической отрасли: 30 040 учёных, 77 центров |

Самое сильное — 12-я редакция Notes of Guidance. Она говорит: **регулятор уже
официально принимает in silico и QSAR как доказательство.** Это переводит продукт
из «интересной технологии» в «способ соответствовать требованиям» — другой бюджет
и другой цикл принятия решения. Там же среди новшеств 12-й редакции значатся
*«templates for PBTK model description and parameter verification and analysis»*:
регулятор не просто принимает вычислительную работу, он **просит описание модели и
верификацию параметров**. Дозье с документированным разделом методов говорит на
языке ассессора; дозье с ранжированным списком и чёрным ящиком за ним — нет.

**Граница, которую нельзя переходить.** Регламент 655/2013 требует, чтобы
in-vitro и ex-vivo были «predictive of an action or representative of an in-vivo
effect, **but studies on humans must validate these predictive effects**».
Обзорная литература фиксирует статус прямо: in silico применяется «predominantly
for **internal, rather than regulatory, decision making**». Вы продаёте внутреннюю
опору для решений и снижение риска до траты денег, а не регуляторное одобрение.
Заявить второе — быть дисквалифицированным первым же компетентным safety
assessor. Сказать это первым — маркер компетентности, ровно в интонации ваших
юридических страниц.

Часть регуляторных счётчиков взята из отраслевых сводок, а не из текста
Официального журнала. Перед публикацией сверить с первоисточником: 1 766 позиций
Annex II, 14 заключений SCCS, €2.35 млрд.

---

## Предлагаемая структура: 7 секций → 12

| # | Секция | Статус |
|---|---|---|
| 01 | Hero — добавить третий абзац про три слоя, вторую CTA на каталог | правка |
| 02 | Trust strip — происхождение методов и научный совет, с оговоркой | новая |
| 03 | The arithmetic — исправить три числа, сменить 180× | исправить |
| 04 | **The stack** — три слоя | новая |
| 05 | **The instruments** — тизер каталога | новая |
| 06 | **The protocols** — пять стадий в анатомии playmolecule | новая |
| 07 | **One question, end to end** — аквапорин через все стадии; заменяет пустой «The department» | новая |
| 08 | November 2026 — оставить, добавить ссылку на регламент и Notes of Guidance | оставить |
| 09 | **Provenance** — методы с DOI плюс честная строка о том, чего не публиковали | новая |
| 10 | What we sell — добавить «typical input / question to resolve» | правка |
| 11 | **Access** — уровни входа вместо одной формы | новая |
| 12 | Footer — добавить колонку «Proof» | правка |

---

## Порядок работ

1. **Исправить три числа** — полдня, только копирайт, блокирует остальное.
2. **Написать каталог из 24 позиций** — 2–3 дня; требует решения по четырём
   дублирующимся модулям движка.
3. **Добавить Technology stack и примеры запросов** — 1–2 дня; наивысшая отдача
   на слово.
4. **Переписать FAQ про метод и завести страницу Methods** — час на FAQ, 3–4 дня
   на Methods по образцу Axiom: базы, движки, скоринги, регуляторные списки, шаги
   QC. SCCS прямо просит описание модели и верификацию параметров, так что
   Methods — это ещё и предварительный ответ ассессору.
5. **Провести один вопрос через все пять стадий** — 1–2 недели, нужен научный
   сотрудник; ретроспективно на публичной молекуле, клиент не нужен. Форма по
   образцу Deep Origin: базовая линия из литературы с DOI, воронка в числах,
   метрика новизны по Танимото, гейт «подробности под NDA», плюс одна карточка
   Stop / Advance с рассуждением. Самый конвертирующий артефакт из всего списка.
6. **Починить движок перед публичными ссылками на него** — неделя; предусловие
   для любого публичного демо.
7. **Свободный вход** — после остального; продуктовое решение, не страница.

---

## Чего делать не надо

- **Не строить интерактивное демо ради плотности.** У playmolecule его нет.
- **Не ставить движок проницаемости в центр.** Это одна позиция каталога из
  двадцати четырёх, внутри стадии Prove. Страница вокруг него сузит продукт до
  «калькулятора проницаемости».
- **Не публиковать бенчмарки, которых нет.** playmolecule предлагает валидацию
  *как услугу* («benchmark it on your compounds, four to six weeks»), а не
  показывает её.
- **Не терять интонацию юридических страниц.** Каталог должен быть написан тем же
  голосом, иначе прочитается как маркетинг.
- **Не ставить круглых непроверяемых чисел.** Debut пишет на главной «100% success
  from AI generation to validated molecule» — без знаменателя и без определения
  «validated». Научный покупатель дисконтирует такое быстрее всего; ваши «8
  специалистов» и «≈10⁵⁰» сейчас из этой же категории.
- **Не ждать логотипов, чтобы начать.** Именные отзывы с должностями работают не
  хуже: у Skinobs это Dior, Sisley и Clarins поимённо, у Potion — Olaplex и THG
  Labs. Один разговор с пилотным клиентом даёт то, чего не даст год ожидания права
  поставить логотип.
