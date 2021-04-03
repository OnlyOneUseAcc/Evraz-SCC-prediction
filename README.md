# Легирование сталей
# Прогнозирование химического состава шлака
 В данной работе мы изучили данные физико-химического процесса легирования сталей и создали алгоритм определения химического состава шлака по исходным данным.
## План работ
### 1. Сделать EDA (exploratory data analysis):
 * Пропуски
 * Нули
 * Выбросы
 * Дисперсия

### 2. Чистка датасета 
* Убрать ненужные столбцы:
    * Несущие бесполезную информацию (Например, признак "nplv")
    * Сильно коррелирующие с другими (Например, признаки "t обработка" и "t продувка"; "t под током" и "эл. энергия")
* Удалить строки содержащие малое количество информации
* Удалить строки, в которых значения таргета неизвестно
* Удаление шумов (выбросов) с помощью IsolationForest

### 3. Заполнение пропущенных значений
* Заполнение с помощью метода MICE (Multiple Imputation by Chained Equations)

### 4. Нормализация вещественных признаков
### 5. Разбиение датасета  на обучающую и тестовую выборки
### 6. Выбор модели для прогнозирования
* Нами было принято решение использовать GradientBoostingRegressor

### 7. Подбор гиперпараметров модели
* Подбор гиперпаметров происходил по n_estimators и lr (learning rate). Эти параметры являются важнейшими для GBR
### 8. Обучение модели на обучающей выборке
* Мы обучали 4 модели для каждого таргета. Для этого создали 4 датасета, каждый из которых включал в себя помимо первоначальных признаков, 3 оставшихся таргета. Мы выбрали данный подход, так как заметили сильную корреляцию между целевыми переменными на этапе обработки данных.
### 9. Получение предсказаний модели на тестовой выборке по самой популярной марки
Были получены следующие результаты для таргетов:
* химшлак последний Al2O3
   * <img src="https://render.githubusercontent.com/render/math?math=r^{2}\  score = 0.5962414407493948">
* химшлак последний CaO 
  * <img src="https://render.githubusercontent.com/render/math?math=r^{2}\  score = 0.920571372574975">
* химшлак последний R
  * <img src="https://render.githubusercontent.com/render/math?math=r^{2}\  score = 0.956619985481036">
* химшлак последний SiO2
  * <img src="https://render.githubusercontent.com/render/math?math=r^{2}\  score = 0.9478195883188777">
## Описание репозитория
Репозиторий содержит в себе ноутбуки EDA.ipynb и GBR.ipynb. EDA.ipynb содержит в себе выполнение пунктов  1-6, GBR.ipynb выполнение пунктов 7-9.
Каталог data содержит исходный датасет, и полученные в результате выполнения EDA.ipynb обучающую и тестовую выборки.
Каталог source содержит графики, полученные при анализе данных.
Каталог models содержит сохраненные модели, для воспроизведения результата.
## Выводы по EDA
* График распределения целевой переменной "химшлак последний Al2O3"
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/target_range_1.png?token=AHSE7JNN4FA5NOTLW4U6ZPTANDSZI "Al2O3")
* График распределения целевой переменной "химшлак последний CaO"
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/target_range_2.png?token=AHSE7JM4VV2CRTCMPU6XYQLANDPEY "CaO")
* График распределения целевой переменной "химшлак последний R"
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/target_range_3.png?token=AHSE7JKEB7C6GQCFTL2OBNDANDPHM "R")
* График распределения целевой переменной "химшлак последний SiO2"
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/target_range_4.png?token=AHSE7JN42MFVD3J4NNV6GWLANDPLK "SiO2")
* График корреляций между признаками
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/correletion_map.png?token=AHSE7JJGHDMRFKZYUUKQ54TANDPO4 "Correlation")
* График корреляции между таргетами
![alt-текст](https://raw.githubusercontent.com/OnlyOneUseAcc/Evraz-SCC-prediction/master/source/correletion_map_target.png?token=AHSE7JOD3XWUICJVHTDET7DANDQGK "Correlation")
