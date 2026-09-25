import polars as pl, time
from ber import candidates, prune, config, matcher, pipeline
m = prune.load_model()
import shutil; shutil.rmtree(config.path("cands","dry","_").parent, ignore_errors=True)
for i,p in enumerate(sorted(config.path("cands","prune_sample").glob("*.parquet"))):
    u = pl.read_parquet(p)
    prune.prune(u, m, 12).write_parquet(config.path("cands","dry",f"{i:03d}.parquet"))
print("pruned sample pairs:", candidates.load("dry",["x"]).height)
matcher.make_sets("dry", n_train_x=120_000)
t=time.time()
matcher.featurize("train","dry",pl.read_parquet(config.path("sets","train_x.parquet")),"train_x")
matcher.featurize("train","dry",pl.read_parquet(config.path("sets","eval_x.parquet")),"eval_x")
print("featurize", round(time.time()-t))
ev = matcher.train_and_eval(rounds1=150, rounds2=150, tag="dry")
pipeline.evaluate_preds(ev, "dry")
