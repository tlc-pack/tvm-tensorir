"""
pytest Extra Options
"""

def pytest_addoption(parser):
    parser.addoption('--nvprof', action="store_true", default=False)
    parser.addoption('--template', action="store", type=str)
    parser.addoption('--filename', action="store", type=str, default="")
    parser.addoption('--force_overwrite', action="store_true", default=False)
    """
    Dense
    """
    parser.addoption('-B', action="store", type=int, default=16)
    parser.addoption('-T', action="store", type=int, default=64)
    parser.addoption('-I', action="store", type=int, default=768)
    parser.addoption('-H', action="store", type=int, default=2304)

    parser.addoption('--dyT', action="store_true", default=False)
    parser.addoption('--dyI', action="store_true", default=False)
    parser.addoption('--dyH', action="store_true", default=False)
