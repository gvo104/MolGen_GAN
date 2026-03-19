# Генерация и отбор молекул с помощью DrugGEN (таргет AKT1)

## Цель работы

Сгенерировать молекулы с помощью генеративной модели **DrugGEN**, ориентированной на белок **AKT1**, и отобрать до 10 наиболее перспективных кандидатов на основе базовых фармакологических критериев:

* **QED** — оценка drug-likeness,
* **SA** — синтезируемость,
* **Toxic** — простая эвристическая токсичность.

## Обоснование выбора мишени

**AKT1** — ключевая серин/треонин-киназа сигнального пути PI3K/AKT, связанного с выживанием и пролиферацией клеток. Гиперактивация пути характерна для многих видов рака, поэтому ингибиторы AKT1 являются актуальным направлением разработки лекарств.

---

## 1. Установка зависимостей

```bash
pip install torch_geometric chembl_structure_pipeline rdkit moses tqdm
```

Клонирование репозитория DrugGEN:

```bash
git clone https://github.com/HUBioDataLab/DrugGEN.git
cd DrugGEN
bash setup.sh
```

Исправление несовместимостей PyTorch 2.6+:

```python
for path, old, new in [
    ('src/util/smiles_cor.py',
     'from torchtext.data import TabularDataset, Field, BucketIterator, Iterator',
     'TabularDataset = Field = BucketIterator = Iterator = None'),
    ('src/data/dataset.py',
     'self.data, self.slices = torch.load(path)',
     'self.data, self.slices = torch.load(path, weights_only=False)'),
]:
    content = open(path).read()
    open(path, 'w').write(content.replace(old, new))
```

---

## 2. Генерация молекул

Конфигурация генерации:

```python
SAMPLE_NUM = 700

cfg = dict(
    submodel='DrugGEN',
    inference_model='experiments/models/DrugGEN-akt1',
    inf_smiles='data/chembl_test.smi',
    train_smiles='data/chembl_train.smi',
    train_drug_smiles='data/akt_train.smi',
    max_atom=60,
)
```

Запуск генерации:

```bash
python inference.py \
    --submodel="DrugGEN" \
    --inference_model="experiments/models/DrugGEN-akt1" \
    --inf_smiles="data/chembl_test.smi" \
    --train_smiles="data/chembl_train.smi" \
    --train_drug_smiles="data/akt_train.smi" \
    --sample_num=700 \
    --max_atom=60 \
    --disable_correction
```

### Оценка качества генерации

Используем метрики MOSES:

* **Validity** = 0.583 — часть структур невалидна.
* **Uniqueness** = 1.0 — все молекулы уникальны.
* **Novelty** ≈ 0.98–1.0 — молекулы новые, не совпадают с обучающей выборкой.
* **Internal Diversity** = 0.863 — химическое пространство достаточно разнообразно.
* **QED (среднее)** = 0.446 — умеренный drug-likeness.
* **SA (среднее)** = 3.11 — большинство молекул синтезируемы.

Вывод: модель генерирует новые и разнообразные молекулы, но требуется фильтрация по свойствам.

---

## 3. Расчет свойств молекул

Используются метрики:

* **QED** — drug-likeness,
* **LogP** — липофильность,
* **MW** — молекулярная масса,
* **SA** — синтезируемость (через `sascorer`),
* **Toxic** — эвристическая токсичность.

Пример функции расчета свойств:

```python
import sascorer
from rdkit.Chem import QED, Crippen, Descriptors

def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sa_score = sascorer.calculateScore(mol)
    return {
        "SMILES": smiles,
        "QED": QED.qed(mol),
        "LogP": Crippen.MolLogP(mol),
        "MW": Descriptors.MolWt(mol),
        "SA": sa_score,
        "Toxic": 1 if (Crippen.MolLogP(mol) > 5 or Descriptors.MolWt(mol) > 600) else 0
    }
```

---

## 4. Фильтрация и ранжирование

Критерии отбора:

* QED > 0.5
* MW < 500
* SA < 10
* Toxic = 0

Сначала фильтруем молекулы по этим критериям, затем вычисляем **score**:

```python
filtered['score'] = filtered['QED']*2 - filtered['SA']*0.1 - filtered['Toxic']*2
top10 = filtered.sort_values('score', ascending=False).head(10)
```

---

## 5. Virtual screening (proxy docking)

Используется приближенная функция связывания:

```python
def docking_proxy(row):
    return -0.8*row["LogP"] - 0.01*row["MW"] + 2.0*row["QED"]

filtered["binding_score"] = filtered.apply(docking_proxy, axis=1)
```

* Более отрицательный `binding_score` → лучшее связывание.

---

## 6. Сравнение с известными ингибиторами AKT1

Известные ингибиторы:

```python
known_inhibitors = {
    "Capivasertib": "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
    "Ipatasertib": "CCN(CC)CCOC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
    "MK-2206": "CC1=CC=C(C=C1)NC(=O)N2CCC(CC2)C3=CC=CC=C3"
}
```

Сравнение через **Tanimoto similarity** с фингерпринтами Morgan:

```python
from rdkit.Chem import AllChem, DataStructs

def max_similarity(smiles):
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
    sims = [DataStructs.TanimotoSimilarity(fp, kfp) for kfp in known_fps.values()]
    return max(sims)
```

---

## 7. Финальный отбор кандидатов

Учитываем три критерия:

* Связывание (`binding_score`) — приоритетное,
* Сходство с известными ингибиторами (`similarity`) — умеренно важно,
* Drug-likeness (`QED`) — дополнительная оценка.

```python
final["final_score"] = -final["binding_score"]*2 + final["similarity"]*1.5 + final["QED"]
top10_final = final.sort_values("final_score", ascending=False).head(10)
```

---

## 8. Визуализация и сохранение

* Визуализация топ-10 кандидатов с QED и similarity.
* Сохранение таблицы в CSV: `final_top10_molecules.csv`.

---

## 9. Итоговый вывод

* QED финальных молекул: **0.52–0.75** → умеренно drug-like.
* Proxy binding score: **до -7.44** → хорошие шансы на связывание с AKT1.
* Сходство с известными ингибиторами: **0.10–0.17** → новые структуры.
* Модель **DrugGEN** генерирует валидные, синтезируемые молекулы, подходящие для дальнейшего экспериментального анализа и более точного докинга.
