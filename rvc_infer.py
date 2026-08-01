import argparse
import sys
import os

# Monkeypatch dataclasses to allow mutable defaults (required for fairseq in Python 3.11+)
import dataclasses
_orig_get_field = dataclasses._get_field

def _patched_get_field(cls, name, type, kw_only):
    try:
        return _orig_get_field(cls, name, type, kw_only)
    except ValueError as e:
        if 'mutable default' in str(e):
            val = getattr(cls, name, dataclasses.MISSING)
            f = dataclasses.Field(
                default=val,
                default_factory=dataclasses.MISSING,
                init=True,
                repr=True,
                hash=None,
                compare=True,
                metadata=None,
                kw_only=kw_only
            )
            f.name = name
            f.type = type
            return f
        raise

dataclasses._get_field = _patched_get_field


def main():
    parser = argparse.ArgumentParser(description="RVC Inference CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--index", type=str, required=False, default="", help="Path to .index file")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio")
    parser.add_argument("--output", type=str, required=True, help="Path to output audio")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch adjustment (semitones)")
    parser.add_argument("--method", type=str, default="rmvpe", help="f0 method (rmvpe, pm, harvest)")
    args = parser.parse_args()

    # Import inside main to save time if help/error
    from rvc_python.infer import RVCInference
    import torch
    
    try:
        # Detect device dynamically (cpu or cuda:0)
        device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        
        # Initialize the RVC inference pipeline
        vc = RVCInference(device=device)
        
        # Load the model
        vc.load_model(args.model, index_path=args.index if args.index else "")
        
        # Set parameters
        vc.set_params(
            f0up_key=args.pitch,
            f0method=args.method
        )
        
        # Perform inference
        vc.infer_file(
            input_path=args.input,
            output_path=args.output
        )
        print(f"✅ RVC generation successful: {args.output}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ RVC Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

