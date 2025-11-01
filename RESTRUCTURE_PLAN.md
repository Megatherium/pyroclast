# Project Restructure Plan

## Current Problems

1. **Bloated Root Directory**: 13 Python files in project root
2. **Verbose CLI Code**: argparse is verbose, lots of boilerplate
3. **Scattered Utilities**: Print functions, formatting utilities mixed together
4. **No Package Structure**: Can't easily import/reuse code
5. **Naming Inconsistency**: Mix of `etorch_*`, `jit_*`, etc.

## Proposed Structure

```
pyroclast/
├── pyroclast.py              # Main CLI entry point
├── pyproject.toml            # Package config (PEP 621)
├── README.md
├── QUICKSTART.md
├── pytest.ini
├── .gitignore
├── pyroclast/                # Main package
│   ├── __init__.py
│   ├── cli.py               # Click CLI definitions
│   │
│   ├── commands/            # Click command handlers
│   │   ├── __init__.py
│   │   ├── convert.py       # convert command
│   │   ├── run.py           # run command
│   │   ├── analyze.py       # analyze command
│   │   ├── doctor.py        # doctor command
│   │   ├── compare.py       # compare command
│   │   ├── optimize.py      # optimize command
│   │   └── jit.py           # JIT commands group
│   │
│   ├── core/                # Core business logic
│   │   ├── __init__.py
│   │   ├── converter.py     # ExecutorchConverter class
│   │   ├── runner.py        # ExecutorchRunner class
│   │   ├── analyzer.py      # ModelAnalyzer class
│   │   ├── comparator.py    # ModelComparator class
│   │   ├── optimizer.py     # AutoOptimizer class
│   │   ├── doctor.py        # ModelDoctor class
│   │   └── jit/
│   │       ├── __init__.py
│   │       ├── converter.py # JIT converter
│   │       └── runner.py    # JIT runner
│   │
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── ui.py            # Print utilities, Colors, banners
│       ├── io.py            # InputSpec, load_input_file
│       └── formats.py       # format_bytes, format_latency, etc.
│
├── tests/
│   ├── conftest.py
│   ├── test_utils/
│   │   ├── test_ui.py
│   │   ├── test_io.py
│   │   └── test_formats.py
│   └── test_core/
│       ├── test_converter.py
│       ├── test_optimizer.py
│       └── test_doctor.py
│
├── scripts/                 # Standalone scripts (not part of package)
│   ├── test_simple_model.py
│   └── parler_tts/
│       ├── converter.py
│       └── jit.py
│
├── docs/
│   ├── FUTURE_IDEAS.md
│   └── NIGHT_BUILD_SUMMARY.md
│
├── outputs/
├── models/
└── cache/
```

## Migration Strategy

### Phase 1: Create Package Structure
1. Create `pyroclast/` package directory
2. Create subdirectories: `commands/`, `core/`, `utils/`
3. Add `__init__.py` files

### Phase 2: Migrate Utilities
1. Split `etorch_utils.py` into:
   - `pyroclast/utils/ui.py` - Colors, print_* functions
   - `pyroclast/utils/io.py` - InputSpec, load_input_file
   - `pyroclast/utils/formats.py` - format_bytes, format_latency

### Phase 3: Migrate Core Classes
1. `etorch_converter.py` → `pyroclast/core/converter.py`
2. `etorch_runner.py` → `pyroclast/core/runner.py`
3. `model_analyzer.py` → `pyroclast/core/analyzer.py`
4. `etorch_compare.py` → `pyroclast/core/comparator.py`
5. `etorch_optimizer.py` → `pyroclast/core/optimizer.py`
6. `etorch_doctor.py` → `pyroclast/core/doctor.py`
7. `jit_converter.py` → `pyroclast/core/jit/converter.py`
8. `jit_runner.py` → `pyroclast/core/jit/runner.py`

### Phase 4: Migrate to Click
1. Install Click: `uv pip install click`
2. Create `pyroclast/cli.py` with Click command groups
3. Create individual command handlers in `pyroclast/commands/`
4. Each command imports and uses core classes

### Phase 5: Update Entry Point
1. Update `pyroclast.py` to use Click CLI
2. Much simpler than current argparse version

### Phase 6: Update Tests
1. Update imports in tests
2. Add new tests for reorganized modules

### Phase 7: Cleanup
1. Remove old files from root
2. Update documentation
3. Update .gitignore if needed

## Click CLI Example

**Before (argparse - verbose):**
```python
def main():
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="command")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("model_path")
    convert_parser.add_argument("-o", "--output-dir", default="./outputs")
    # ... 20 more lines
```

**After (Click - concise):**
```python
import click

@click.group()
def cli():
    """Pyroclast - Executorch Toolkit"""
    pass

@cli.command()
@click.argument('model_path')
@click.option('-o', '--output-dir', default='./outputs')
@click.option('-b', '--backend', type=click.Choice(['xnnpack', 'vulkan', 'portable']))
def convert(model_path, output_dir, backend):
    """Convert model to Executorch format"""
    from pyroclast.core.converter import ExecutorchConverter
    converter = ExecutorchConverter(model_path, output_dir, backend)
    converter.convert()
```

## Benefits

### Developer Experience
- **Cleaner imports**: `from pyroclast.utils.ui import print_success`
- **Better IDE support**: Autocomplete, type hints
- **Easier testing**: Import specific classes/functions
- **Clear separation**: UI, core logic, utilities

### User Experience
- **Cleaner CLI**: Click provides better help, autocomplete
- **Subcommand groups**: `pyroclast jit convert` instead of `pyroclast jit-convert`
- **Better errors**: Click gives helpful error messages

### Maintenance
- **Package structure**: Can be installed with `pip install -e .`
- **Modular**: Easy to add new commands/features
- **Testable**: Each module independently testable
- **Professional**: Follows Python packaging best practices

## Migration Checklist

- [ ] Install Click
- [ ] Create package structure
- [ ] Split etorch_utils.py into ui/io/formats
- [ ] Migrate core classes to pyroclast/core/
- [ ] Create Click CLI in pyroclast/cli.py
- [ ] Create command handlers in pyroclast/commands/
- [ ] Update pyroclast.py entry point
- [ ] Update all tests
- [ ] Update README.md examples
- [ ] Remove old files
- [ ] Test all commands work
- [ ] Update documentation

## Estimated Time

- **Phase 1-3** (Package + Utils + Core): 1-1.5 hours
- **Phase 4-5** (Click migration): 1 hour
- **Phase 6-7** (Tests + Cleanup): 30 minutes

**Total**: ~2.5-3 hours

## Decision Points

### Should we use Click?
**YES** - Benefits far outweigh migration cost:
- Much cleaner code (50% less boilerplate)
- Better UX (help text, autocomplete, validation)
- Industry standard (used by Flask, Pytest, AWS CLI)
- Easy subcommand groups

### Should we make it a proper package?
**YES** - Even if not published to PyPI:
- Better code organization
- Can be installed in editable mode
- Follows Python best practices
- Makes future growth easier

### Keep old scripts for reference?
**NO** - Git history preserves them:
- Clean migration, no half-measures
- Can always reference old commits if needed

## Next Steps

1. Get user approval on plan
2. Create feature branch: `feat/project-restructure`
3. Execute phases 1-7
4. Test thoroughly
5. Update documentation
6. Merge to WIP

---

**Ready to execute!** 🚀
