"""Find openenv validate logic by examining the CLI entry point."""
import importlib.metadata as md
import inspect, importlib

# Get all entry points for openenv-core
dist = md.distribution('openenv-core')
print("=== Package files ===")
for f in dist.files[:20]:
    print(f"  {f}")

print("\n=== Entry points ===")
for ep in dist.entry_points:
    print(f"  {ep.name} = {ep.value} (group={ep.group})")
    if ep.group == 'console_scripts':
        try:
            mod_name, func_name = ep.value.split(':')
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            src = inspect.getsource(func)
            print(f"\n=== SOURCE OF {ep.value} ===")
            for i, line in enumerate(src.split('\n')):
                print(f"  {i}: {line}")
                if i > 100:
                    print("  ... (truncated)")
                    break
        except Exception as e:
            print(f"  Could not load: {e}")
            # Try loading the module source
            try:
                mod = importlib.import_module(mod_name)
                src = inspect.getsource(mod)
                for i, line in enumerate(src.split('\n')):
                    ll = line.lower()
                    if any(kw in ll for kw in ['script', 'pyproject', 'validate', 'server', 'main', 'fail', 'issue', 'missing', 'not ready']):
                        print(f"  L{i}: {line}")
            except Exception as e2:
                print(f"  Also failed: {e2}")
