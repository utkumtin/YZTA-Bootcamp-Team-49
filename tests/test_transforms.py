import pandas as pd
import pytest

from pareto.cleaning.transforms import apply_transform, get_transform


def test_arbitrary_transform_rejected():
    # NEDEN: L3 güvenlik — kapalı taksonomi dışı hiçbir şey seçilemez. Prototipteki
    # keyfi-kod/exec yolu bu testle kapalı tutulur (RCE regresyonu yakalanır).
    with pytest.raises(ValueError):
        get_transform("os_system")


def test_coerce_numeric_strips_thousands_separator():
    df = pd.DataFrame({"x": ["1,234", "5,678"]})
    out = apply_transform(df, "coerce_numeric", {"col": "x"})
    assert out["x"].tolist() == [1234, 5678]


def test_render_returns_code_string_not_execution():
    # NEDEN: transform kod DÖNDÜRÜR (denetim izi), çalıştırmaz; exec yolu yok.
    code = get_transform("rename_column").render(old="a", new="b")
    assert "rename" in code
    assert "exec" not in code
