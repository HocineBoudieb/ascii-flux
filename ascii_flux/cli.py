"""Small explicit CLI; optional training dependencies are imported on demand."""
import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="ASCII Flux v5 research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("convert", help="Offline local YouTube-VIS -> ASCII shards")
    for name in ("annotations", "frames-root", "output"):
        build.add_argument("--"+name, required=True)
    build.add_argument("--source", default="configs/ytvis_source.json")
    build.add_argument("--config", default="configs/d1.json")
    build.add_argument("--source-fps", required=True, type=float, help="Actual cadence of the extracted annotated frames")
    review = sub.add_parser("review", help="Build standalone D1 review HTML")
    review.add_argument("--manifest", required=True)
    review.add_argument("--output", required=True)
    scale = sub.add_parser("scale", help="Resume bounded D2 shard conversion into SQLite catalogue")
    for name in ("annotations", "frames-root", "source", "output"):
        scale.add_argument("--"+name, required=True)
    scale.add_argument("--config", default="configs/d2.json")
    scale.add_argument("--source-fps", type=float, required=True)
    scale.add_argument("--workers", type=int, default=2)
    scale.add_argument("--max-jobs", type=int)
    catalog = sub.add_parser("catalogue", help="Index, count or freeze existing ASCII shard runs")
    catalog.add_argument("--root", required=True)
    catalog.add_argument("--ingest")
    catalog.add_argument("--snapshot", help="Immutable manifest filename in catalogue root")
    million = sub.add_parser("million", help="Acquire, convert, freeze and train one million grid slots on local GPU")
    million.add_argument("--project-root", default=".")
    million.add_argument("--data-root")
    million.add_argument("--workers", type=int, default=4)
    embed = sub.add_parser("embed", help="Cache pinned frozen SigLIP 2 text embeddings")
    embed.add_argument("--manifest", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--device", default="cpu")
    training = sub.add_parser("train", help="Train the native glyph NCA baseline")
    for name in ("manifest", "embeddings", "output"):
        training.add_argument("--"+name, required=True)
    training.add_argument("--reviews")
    training.add_argument("--config")
    training.add_argument("--device", default="cpu")
    training.add_argument("--resume")
    training.add_argument("--allow-unreviewed", action="store_true")
    training.add_argument("--allow-fixture", action="store_true")
    for name in ("evaluate", "export", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True, help="Tensor-only best.pt inference artifact")
        p.add_argument("--allow-development", action="store_true")
        p.add_argument("--device", default="cpu")
        if name != "serve":
            p.add_argument("--output", required=True)
        if name == "evaluate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--switch-prompt")
            p.add_argument("--steps", type=int, default=600)
        if name == "serve":
            p.add_argument("--port", type=int, default=8766)
            p.add_argument("--fps", type=float, default=24.)
            p.add_argument("--debounce-ms", type=float, default=0., help="Text delay; zero sends every input revision immediately")
    args = parser.parse_args(argv)
    if args.command == "convert":
        from .factory import build_dataset
        result = build_dataset(args.annotations, args.frames_root, read_json(args.source), read_json(args.config), args.output, args.source_fps)
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    elif args.command == "scale":
        from .scale import build_scale
        result = build_scale(args.annotations, args.frames_root, read_json(args.source), read_json(args.config),
                             args.output, args.source_fps, args.workers, args.max_jobs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "catalogue":
        from .catalogue import Catalogue
        with Catalogue(args.root) as catalogue:
            if args.ingest:
                catalogue.ingest(args.ingest)
            if args.snapshot:
                print(catalogue.snapshot(args.snapshot))
            print(json.dumps(catalogue.report(), ensure_ascii=False, indent=2))
    elif args.command == "million":
        from .pipeline import run_million
        run_million(args.project_root, args.data_root, args.workers)
    elif args.command == "review":
        from .review import review_page
        print(review_page(args.manifest, args.output))
    elif args.command == "embed":
        from .text import cache_embeddings
        print(cache_embeddings(args.manifest, args.output, args.device))
    elif args.command == "train":
        from .training import TrainConfig, train
        train(args.manifest, args.embeddings, args.output, TrainConfig(**read_json(args.config)) if args.config else None,
              reviews_path=args.reviews, device=args.device, resume=args.resume,
              allow_unreviewed=args.allow_unreviewed, allow_fixture=args.allow_fixture)
    elif args.command == "serve":
        from .runtime import serve
        serve(args.checkpoint, args.port, args.fps, args.device, args.allow_development, args.debounce_ms/1000)
    else:
        from .inference import load_inference
        from .evaluation import export_onnx, measure_rollout
        from .storage import write_json
        model, provenance = load_inference(args.checkpoint, args.device, args.allow_development)
        if args.command == "export":
            print(export_onnx(model, args.output))
        else:
            if provenance.get("synthetic_encoder"):
                raise ValueError("Use fixture diagnostics for synthetic text vectors; do not apply real prompts")
            from .text import TextEncoder
            encoder = TextEncoder(args.device)
            prompts = [args.prompt] + ([args.switch_prompt] if args.switch_prompt else [])
            vectors = encoder.encode(prompts)
            result = measure_rollout(model, vectors[0], args.steps, switch_embedding=vectors[1] if len(vectors)>1 else None)
            result["text_encoding_scope"] = "performed once per prompt outside model timing"
            result["checkpoint_provenance"] = provenance
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))

