# SP1 - Vektorový model pro vyhledávání informací

## Zadání
zadání dostupné [zde](./LatexReferat/SZPJ_SP1_zadani.pdf)

## Spuštění programu

### Instalace potřebných knihoven pomocí souboru [requirements.txt](../requirements.txt)
Instalace pomocí použití následujícího příkazu
```commandline
pip install -r requirements.txt
```

### Spuštění hlavního programu
Hlavní program je ve skriptu [main.py](./main.py), kde je zapsán kokmpletní běh nastavný na funkčnost 
na dev datech poskytnutých [zde](./SZPJ_SP1_collection).

Pro případnou úpravu dat je potřeba změnit hyperparametry ve zmíněném [skriptu](./main.py).
> Ukázka defaultních hyperparametrů ve skriptu [main.py](./main.py)
> ```python
> # Hyperparameters
> useStemming = True  # swith for comparing with and without stemming
> document_path = Path('./SZPJ_SP1_collection/documents')  # path to dictionary with source content files
> queries_path = Path('./SZPJ_SP1_collection/query_devel.xml')  # path to file containing queries
> output_path = Path('./vector_search_output.txt')  # savepath for output of search
> qrel_path = Path('./SZPJ_SP1_collection/cacm_devel.rel')  # file for computing score
> ```

Avšak veškeré metody pro výpočet jsou obsažené ve skriptu [vector_search.py](./vector_search.py), 
které jsou pak do [main.py](./main.py) importovány.

## Výsledky
**MAP** na dev datech: **0.3625** (při použití stemmingu) / **0.3277** (bez stemmingu)

## Krátká dokumentace
Krátký popis metod a postupu [zde](./LatexReferat/main.pdf)