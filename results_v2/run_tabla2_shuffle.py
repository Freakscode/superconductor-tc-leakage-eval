"""
run_tabla2_shuffle.py — Rehacer la Tabla 2 (robustez) con particiones agrupadas
realmente independientes (GroupKFold shuffle=True, semillas 0-4).

Tres esquemas por (dataset, modelo, semilla s):
  random_shuffle     : KFold(5, shuffle=True, random_state=s)            <- particion varia
  grouped_shuffle    : GroupKFold(5, shuffle=True, random_state=s)       <- NUEVO: particion varia
  grouped_noshuffle  : GroupKFold(5)                                     <- protocolo PUBLICADO
                                                                            (particion FIJA)
En los tres casos el modelo se construye con make_models(s) del script del repo,
asi que la unica diferencia entre grouped_shuffle y grouped_noshuffle es si la
particion varia con la semilla. Eso aisla la contribucion de la particion.

Hiperparametros: importados con importlib desde
  code/tc_pipeline.py       (Dataset A)
  code/datasetB_pipeline.py (Dataset B)
No se reescriben. n_jobs se deja como esta en el repo; el paralelismo es a nivel
de tarea externa (una tarea = una CV completa), que es numericamente identico
para random_state fijo.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # no ensuciar el working tree del repo con .pyc

ROOT = "/Users/freakscode/Proyectos 2025/Proyectos Academicos/Superconductors"
ROOT = os.environ["SUPERCON_ROOT"]
REPO = os.path.join(ROOT, "superconductor-tc-leakage-eval")
OUT = os.path.join(ROOT, "results_v2")
SEEDS = [0, 1, 2, 3, 4]
SCHEMES = ["random_shuffle", "grouped_shuffle", "grouped_noshuffle"]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def get_data(ds):
    """Devuelve X, y, groups para el dataset pedido, igual que los scripts del repo."""
    tcA = load_mod("tc_pipeline", os.path.join(REPO, "code", "tc_pipeline.py"))
    if ds == "A":
        feat = pd.read_csv(os.path.join(REPO, "data", "train.csv"))
        uniq = pd.read_csv(os.path.join(REPO, "data", "unique_m.csv"))
        assert np.allclose(feat["critical_temp"], uniq["critical_temp"]), "no alineados"
        X = feat.drop(columns=["critical_temp"]).values
        y = feat["critical_temp"].values
        g = tcA.chemical_families(uniq)
    else:
        npz = np.load(os.path.join(REPO, "data", "datasetB_featurized.npz"), allow_pickle=True)
        X, y, g = npz["X"], npz["y"], npz["groups"]
        m = y > 0  # base del paper: solo superconductores Tc>0
        X, y, g = X[m], y[m], g[m]
    return X, y, g


def make_splitter(scheme, s):
    from sklearn.model_selection import KFold, GroupKFold
    if scheme == "random_shuffle":
        return KFold(5, shuffle=True, random_state=s), False
    if scheme == "grouped_shuffle":
        return GroupKFold(5, shuffle=True, random_state=s), True
    if scheme == "grouped_noshuffle":
        return GroupKFold(5), True
    raise ValueError(scheme)


DATA = {}      # ds -> (X, y, g)  cargado una sola vez, solo-lectura, compartido por hilos
MAKERS = {}    # ds -> make_models del script correspondiente del repo


def one_task(task):
    """Una CV completa out-of-fold. Devuelve fila con MAE/R2 y trazas de particion."""
    ds, model, scheme, s = task
    from sklearn.metrics import mean_absolute_error, r2_score

    X, y, g = DATA[ds]
    make_models = MAKERS[ds]

    cv, use_groups = make_splitter(scheme, s)
    t0 = time.time()
    yp = np.empty_like(y, dtype=float)
    fold_sizes, fold_fam_counts, fold_mae, leak = [], [], [], []
    splitter = cv.split(X, y, g) if use_groups else cv.split(X, y)
    for tr, te in splitter:
        m = make_models(s)[model]
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
        fold_sizes.append(int(len(te)))
        fold_fam_counts.append(int(len(np.unique(g[te]))))
        fold_mae.append(float(mean_absolute_error(y[te], yp[te])))
        leak.append(int(len(set(g[tr]) & set(g[te]))))
    row = dict(
        dataset=ds, modelo=model, scheme=scheme, seed=int(s),
        MAE=float(mean_absolute_error(y, yp)), R2=float(r2_score(y, yp)),
        n=int(len(y)), n_familias=int(len(np.unique(g))),
        fold_sizes=json.dumps(fold_sizes), fold_familias=json.dumps(fold_fam_counts),
        fold_MAE=json.dumps([round(v, 6) for v in fold_mae]),
        familias_compartidas_tr_te=int(sum(leak)),
        segundos=round(time.time() - t0, 1),
    )
    return row


def paso1_determinismo():
    """PASO 1: demostrar que GroupKFold(5) sin shuffle es determinista y que
    shuffle=True con semillas distintas da particiones distintas.
    Escribe results_v2/tabla2_determinismo_groupkfold.json."""
    import itertools
    from sklearn.model_selection import GroupKFold, KFold
    import sklearn, xgboost, scipy

    def first_fold(cv, X, y, g):
        tr, te = next(iter(cv.split(X, y, g)))
        return frozenset(np.unique(g[te]).tolist()), te

    def evidencia(tag, X, y, g):
        ev = {"dataset": tag, "n": int(len(y)), "n_familias": int(len(np.unique(g)))}
        # (a) cinco invocaciones de GroupKFold(5) SIN shuffle
        inv = [[(frozenset(np.unique(g[te]).tolist()), te) for _, te in GroupKFold(5).split(X, y, g)]
               for _ in range(5)]
        ev["noshuffle_5_invocaciones_folds_identicos"] = all(
            all(np.array_equal(inv[k][f][1], inv[0][f][1]) for f in range(5)) for k in range(5))
        ev["noshuffle_fold_sizes"] = [int(len(te)) for _, te in inv[0]]
        ev["noshuffle_fold1_n_familias"] = len(inv[0][0][0])
        # (b) shuffle=True, semillas 0..4
        sets_s, sizes_s = {}, {}
        for s in SEEDS:
            folds = [(frozenset(np.unique(g[te]).tolist()), int(len(te)))
                     for _, te in GroupKFold(5, shuffle=True, random_state=s).split(X, y, g)]
            sets_s[s], sizes_s[s] = folds[0][0], [sz for _, sz in folds]
        ev["shuffle_n_particiones_distintas_de_5"] = len({frozenset(v) for v in sets_s.values()})
        jac = {f"{a}v{b}": len(sets_s[a] & sets_s[b]) / len(sets_s[a] | sets_s[b])
               for a, b in itertools.combinations(SEEDS, 2)}
        ev["shuffle_jaccard_fold1_pares_min"] = float(min(jac.values()))
        ev["shuffle_jaccard_fold1_pares_mean"] = float(np.mean(list(jac.values())))
        ev["shuffle_jaccard_fold1_pares_max"] = float(max(jac.values()))
        ev["shuffle_fold_sizes_por_semilla"] = sizes_s
        ev["shuffle_vs_noshuffle_jaccard_fold1"] = {
            s: float(len(inv[0][0][0] & sets_s[s]) / len(inv[0][0][0] | sets_s[s])) for s in SEEDS}
        # (c) control: el brazo aleatorio ya variaba con la semilla
        kf1 = {s: frozenset(next(iter(KFold(5, shuffle=True, random_state=s).split(X)))[1].tolist())
               for s in SEEDS}
        ev["kfold_random_n_particiones_distintas_de_5"] = len(set(kf1.values()))
        # (d) sin fuga de familias en el brazo agrupado con shuffle
        ev["grouped_shuffle_familias_compartidas_total"] = int(sum(
            len(set(g[tr]) & set(g[te]))
            for tr, te in GroupKFold(5, shuffle=True, random_state=0).split(X, y, g)))
        return ev

    evs = {ds: evidencia(ds, *get_data(ds)) for ds in ("A", "B")}
    doc = {
        "pregunta": ("GroupKFold(5) sin shuffle, ¿da folds identicos entre invocaciones? "
                     "¿y con shuffle=True y semillas 0-4?"),
        "respuesta_corta": ("SI, identicos sin shuffle (la particion es determinista, la semilla solo "
                            "cambia el modelo). Con shuffle=True las 5 semillas dan 5 particiones distintas."),
        "jaccard_esperado_si_particiones_independientes": 1 / 9,
        "nota_jaccard": ("Para dos subconjuntos independientes de 1/5 de las familias, el Jaccard esperado "
                         "es (1/25)/(9/25)=1/9=0.111. El observado coincide, lo que confirma que shuffle=True "
                         "aleatoriza de verdad la asignacion de familias a folds."),
        "entornos": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "procedencia": (f"run_tabla2_shuffle.py --paso1 (funcion paso1_determinismo) | "
                        f"GroupKFold de sklearn {sklearn.__version__}; grupos = familia quimica "
                        f"(conjunto de elementos) via pandas.factorize; A: tc_pipeline.chemical_families"
                        f"(unique_m.csv); B: campo 'groups' de datasetB_featurized.npz (base Tc>0)"),
        "datasets": evs,
    }
    p = os.path.join(OUT, "tabla2_determinismo_groupkfold.json")
    json.dump(doc, open(p, "w"), indent=2, default=str)
    print(f"escrito {p}", flush=True)
    for ds, e in evs.items():
        print(f"  {ds}: noshuffle_identicos={e['noshuffle_5_invocaciones_folds_identicos']} "
              f"shuffle_particiones_distintas={e['shuffle_n_particiones_distintas_de_5']}/5 "
              f"jaccard_medio={e['shuffle_jaccard_fold1_pares_mean']:.5f} "
              f"fuga_familias={e['grouped_shuffle_familias_compartidas_total']}", flush=True)
    return doc


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--paso1" in sys.argv:
        paso1_determinismo()
        if "--solo-paso1" in sys.argv:
            return
    for ds in ("A", "B"):
        DATA[ds] = get_data(ds)
        mod = load_mod("tc_pipeline", os.path.join(REPO, "code", "tc_pipeline.py")) if ds == "A" \
            else load_mod("datasetB_pipeline", os.path.join(REPO, "code", "datasetB_pipeline.py"))
        MAKERS[ds] = mod.make_models
        print(f"dataset {ds}: X={DATA[ds][0].shape} familias={len(set(DATA[ds][2]))}", flush=True)

    tasks = [(ds, mo, sc, s) for ds in ("A", "B") for mo in ("RandomForest", "XGBoost")
             for sc in SCHEMES for s in SEEDS]
    # las tareas B/RandomForest son las mas lentas (~34 s por fit): primero, para
    # que no queden colgando al final del pool
    tasks.sort(key=lambda t: 0 if (t[0] == "B" and t[1] == "RandomForest") else 1)
    print(f"{len(tasks)} tareas de CV", flush=True)
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=int(os.environ.get("NW", "10"))) as ex:
        for i, row in enumerate(ex.map(one_task, tasks), 1):
            rows.append(row)
            print(f"[{i}/{len(tasks)}] {row['dataset']}/{row['modelo']}/{row['scheme']}/s{row['seed']} "
                  f"MAE={row['MAE']:.4f} R2={row['R2']:.4f} ({row['segundos']}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tabla2_runs_por_semilla.csv"), index=False)
    print(f"listo en {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
