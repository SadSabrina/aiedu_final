# Online Shoppers Purchasing Intention
![Red-And-Blue-Scrawl-Sale.png](https://ltdfoto.ru/images/2023/08/20/Red-And-Blue-Scrawl-Sale.png)
## Description 
В проекте представлено решение задачи прогнозирования выручки на основе применения машинного обучения. \
В репозитории содержатся:
- разведовательный анализ данных об активности пользователей на сайте — `Shoppers.ipynb`.
- процессы обучения, тестирования, тюнинга и интерпретации модели машнного обучения — `Model_learning.ipynb`.
- файлы для деплоя и развертывания интерактивного дашборда, позволяющего отследить процесс принятия решений моделью — `app_files`.

## EDA
В процессе разведочного анализа исследовано влияние категориальных и непрерывных признаков на целевую переменную. Для каждого признака статистическими методами проверены гипотезе о его значимости. Выдвинута гипотеза о возможности обучения алгоритма машинного обучения на скоращенном множестве признаков.

## Model learning
В ходе обучения были протестированы 3 вида моделей. Для лучшей достингуто качество `0.82` по метрике `balanced accuracy`. Данная метрика позволяет сделать оценку решения наиболее объективной, поскольку работа проводится в условиях проблемы дисбаланса классов. 

## Solution Benefits

- Множество признаков сокращено с 17 до 6 наиболее значимых без потери качества модели.
- На основе дашборда может быть прослежен процесс принятия решений моделью для каждого наблюдения.
- Достигнут баланс в качестве прогнозирования как бОльшего, так и меньшего классов.

Автор: [Sabrina Sadiekh (telegram)](https://t.me/sabrina_sadiekh) \
Блог автора об интерпретируемом машинном обучении: [Datablog](https://t.me/jdata_blog)

Работа выполнена в рамках курса: [Разведочный анализ данных и основы разработки](https://stepik.org/course/177213/syllabus)


