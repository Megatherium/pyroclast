# Executorch Toolkit - Quick Start Guide

## 🎯 What Is This?

A comprehensive toolkit for converting PyTorch models to Executorch format, with tools for inference, benchmarking, and analysis.

## 🚀 Quick Start (60 seconds)

### 1. Activate Environment

```bash
source ~/.bashrc.d/pyenv.bash
pyenv activate torch
```

### 2. Try the Master CLI

```bash
python3 etorch.py
```

You'll see a beautiful banner and help menu!

### 3. Test with Simple Model

```bash
# Test conversion (takes ~1 minute, downloads model)
python3 test_simple_model.py

# Run benchmarks on converted model
python3 etorch.py run outputs/mobilenet_v2_test.pte --benchmark --runs 50

# Compare performance
python3 etorch.py compare --baseline mobilenet_v2 --executorch outputs/mobilenet_v2_test.pte
```

### 4. Try ParlerTTS Analysis

```bash
# Analyze the complex ParlerTTS model
python3 parler_tts_converter.py models/eclipse_code -o outputs/

# This will explain why ParlerTTS doesn't work with Executorch
# and provide alternatives (ONNX, TorchScript, etc.)
```

## 📚 Main Commands

### Convert a Model

```bash
python3 etorch.py convert models/my_model -o outputs/ --backend xnnpack
```

### Run Inference

```bash
python3 etorch.py run outputs/model.pte --benchmark --runs 100
```

### Analyze Model

```bash
python3 etorch.py analyze models/my_model --max-depth 4
```

### Compare Performance

```bash
python3 etorch.py compare \\
  --baseline mobilenet_v2 \\
  --executorch outputs/*.pte \\
  --save comparison.json
```

## 📖 Documentation

- **README.md**: Comprehensive user guide with all features
- **WRITEUP.md**: Technical deep dive and project retrospective
- **CLAUDE.md**: Project context and achievements
- **This file**: Quick start guide

## 🎨 What Makes This Cool?

✨ **Beautiful CLI**: Colors, ASCII art, progress bars
🚀 **Fast Setup**: Works out of the box
📊 **Rich Metrics**: P95, P99, throughput, comparisons
🔍 **Deep Analysis**: Model architecture inspection
💡 **Smart Guidance**: Explains when conversion won't work
📦 **Complete Toolkit**: Convert, run, analyze, compare

## ⚡ Performance Examples

**MobileNetV2 Benchmarks:**
- PyTorch: 19.39ms/inference (51.57/sec)
- Executorch: 1390ms/inference (0.72/sec) *portable backend*

*XNNPACK backend would be ~10-15ms (faster than PyTorch)*

## 🎯 Files You Created

**Core Tools:**
- `etorch.py` - Master CLI ⭐
- `etorch_converter.py` - Model converter
- `etorch_runner.py` - Inference runner
- `etorch_compare.py` - Performance comparator
- `model_analyzer.py` - Architecture analyzer
- `parler_tts_converter.py` - ParlerTTS handler

**Documentation:**
- `README.md` - User guide
- `WRITEUP.md` - Technical writeup
- `QUICKSTART.md` - This file
- `CLAUDE.md` - Project summary

**Generated:**
- `outputs/mobilenet_v2_test.pte` - Converted model
- `outputs/parler_tts_inference.py` - Reference code
- `outputs/parler_tts_metadata.json` - Model metadata
- `outputs/comparison.json` - Benchmark data

## 🐛 Troubleshooting

### Import Errors

```bash
# Make sure environment is activated
source ~/.bashrc.d/pyenv.bash && pyenv activate torch

# Verify Executorch is available
python3 -c "import executorch; print(executorch.__version__)"
```

### Model Not Found

```bash
# Model is in models/eclipse_code/
ls -lah models/eclipse_code/
```

### Slow Performance

- Portable backend is slow (expected)
- Use XNNPACK for better CPU performance
- Vulkan requires GPU setup

## 💡 Tips

1. **Start simple**: Test with MobileNetV2 before complex models
2. **Read the errors**: Helpful messages explain what went wrong
3. **Check metadata**: Converter creates `.json` files with details
4. **Use --verbose**: Get detailed output for debugging
5. **Save results**: Use `--save` to export JSON for later analysis

## 🎓 Learn More

**Want to understand the tech?**
Read `WRITEUP.md` for a 10-page deep dive covering:
- Executorch pipeline internals
- Why ParlerTTS doesn't work
- Benchmark methodology
- Design decisions
- Lessons learned

**Want to use the tools?**
Read `README.md` for complete documentation on:
- All command-line options
- Backend comparisons
- Example workflows
- Troubleshooting guide

## ✨ Show Off Your Results

```bash
# Generate a cool comparison
python3 etorch.py compare \\
  --baseline mobilenet_v2 \\
  --executorch outputs/mobilenet_v2_test.pte \\
  --runs 200 \\
  --save my_benchmark.json

# Analyze architecture with deep tree
python3 etorch.py analyze models/eclipse_code --max-depth 5 --save-report
```

Take screenshots of the beautiful terminal output! 📸

---

**Have fun exploring Executorch!** 🚀

Built with ❤️ overnight using Claude Code
