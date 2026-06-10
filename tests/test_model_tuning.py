from src.tune_selected_models import _is_better, _metrics, _rank


def test_binary_tuning_rank_prioritizes_f2():
    stronger_f2 = {"F2": 0.8, "F1": 0.7, "Accuracy": 0.7}
    stronger_accuracy = {"F2": 0.79, "F1": 0.8, "Accuracy": 0.9}
    assert _rank(stronger_f2, "binary") > _rank(stronger_accuracy, "binary")


def test_multiclass_tuning_rank_prioritizes_macro_f1():
    stronger_macro = {"Macro F1": 0.4, "Weighted F1": 0.5, "Accuracy": 0.5}
    stronger_accuracy = {"Macro F1": 0.39, "Weighted F1": 0.7, "Accuracy": 0.8}
    assert _rank(stronger_macro, "multiclass") > _rank(
        stronger_accuracy, "multiclass"
    )


def test_binary_metrics_use_informative_as_positive_class():
    values = _metrics(
        ["informative", "not_informative"],
        ["informative", "informative"],
        "binary",
    )
    assert values["Recall"] == 1.0
    assert values["Precision"] == 0.5


def test_multiclass_tie_uses_weighted_f1_within_tolerance():
    incumbent = {"Macro F1": 0.3300, "Weighted F1": 0.4900, "Accuracy": 0.46}
    tiny_macro_gain = {
        "Macro F1": 0.3305,
        "Weighted F1": 0.4850,
        "Accuracy": 0.47,
    }
    assert not _is_better(tiny_macro_gain, incumbent, "multiclass")
