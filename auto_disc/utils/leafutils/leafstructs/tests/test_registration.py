from auto_disc.utils.leafutils.leafstructs.registration import (
    get_cls_from_name,
    get_cls_from_path,
    get_custom_modules,
    get_default_modules,
    get_legacy_modules,
    get_modules,
    get_path_from_cls,
    locate_cls,
)


def test_locate_cls():
    from auto_disc.auto_disc.explorers import IMGEPFactory as cls

    cls_path = "auto_disc.auto_disc.explorers.IMGEPFactory"
    assert cls == locate_cls(cls_path)
    from adtool_default.maps.LeniaStatistics import LeniaStatistics as cls

    cls_path = "adtool_default.maps.LeniaStatistics.LeniaStatistics"
    assert cls == locate_cls(cls_path)


def test_get_cls_from_path():
    # test legacy path
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    cls = get_cls_from_path(path)
    from auto_disc.legacy.explorers.imgep_explorer import IMGEPExplorer as compare_cls

    assert cls == compare_cls

    path = "adtool_default.maps.LeniaStatistics.LeniaStatistics"
    cls = get_cls_from_path(path)
    from adtool_default.maps.LeniaStatistics import LeniaStatistics as compare_cls

    assert cls == compare_cls

    path = "adtool_default.maps.MeanBehaviorMap.MeanBehaviorMap"
    cls = get_cls_from_path(path)
    from adtool_default.maps.MeanBehaviorMap import MeanBehaviorMap as compare_cls

    assert cls == compare_cls


def test_get_path_from_cls():
    from auto_disc.legacy.explorers.imgep_explorer import IMGEPExplorer as compare_cls

    compare_path = get_path_from_cls(compare_cls)
    path = "auto_disc.legacy.explorers.imgep_explorer.IMGEPExplorer"
    assert compare_path == path

    from adtool_default.maps.LeniaStatistics import LeniaStatistics as compare_cls

    compare_path = get_path_from_cls(compare_cls)
    # here we see that `get_path_from_cls` gives an explicit FQDN
    # instead of using imports from `__init__.py` files
    path = "adtool_default.maps.LeniaStatistics.LeniaStatistics"
    assert compare_path == path

    from adtool_default.maps.MeanBehaviorMap import MeanBehaviorMap as compare_cls

    compare_path = get_path_from_cls(compare_cls)
    path = "adtool_default.maps.MeanBehaviorMap.MeanBehaviorMap"
    assert compare_path == path


def test_get_cls_from_name():
    from adtool_default.maps.LeniaStatistics import LeniaStatistics
    from adtool_default.systems.ExponentialMixture import ExponentialMixture
    from auto_disc.auto_disc.explorers import IMGEPFactory
    from auto_disc.auto_disc.utils.callbacks.on_save_callbacks.save_leaf_callback import (
        SaveLeaf,
    )
    from auto_disc.auto_disc.utils.callbacks.on_save_finished_callbacks.generate_report_callback import (
        GenerateReport,
    )

    cls_name = "IMGEPExplorer"
    ad_type_name = "explorers"
    assert get_cls_from_name(cls_name, ad_type_name) == IMGEPFactory

    cls_name = "LeniaStatistics"
    ad_type_name = "maps"
    assert get_cls_from_name(cls_name, ad_type_name) == LeniaStatistics

    cls_name = "ExponentialMixture"
    ad_type_name = "systems"
    assert get_cls_from_name(cls_name, ad_type_name) == ExponentialMixture

    cls_name = "base"
    ad_type_name = "callbacks.on_saved"
    assert get_cls_from_name(cls_name, ad_type_name) == SaveLeaf

    cls_name = "base"
    ad_type_name = "callbacks.on_save_finished"
    assert get_cls_from_name(cls_name, ad_type_name) == GenerateReport


def test_get_legacy_modules():
    assert get_legacy_modules("systems") == {}
    assert get_legacy_modules("explorers").keys() == set(["IMGEPExplorer"])
    assert get_legacy_modules("maps") == {}
    assert get_legacy_modules("callbacks").keys() == set(
        [
            "on_discovery",
            "on_cancelled",
            "on_error",
            "on_finished",
            "on_saved",
            "on_save_finished",
            "interact",
        ]
    )


def test_get_custom_modules():
    # by default, there are no custom modules
    assert get_custom_modules("systems") == {}
    assert get_custom_modules("explorers") == {}
    assert get_custom_modules("maps") == {}
    assert get_custom_modules("callbacks") == {}


def test_get_default_modules():
    assert set(get_default_modules("systems").keys()) == set(
        ["ExponentialMixture", "Lenia", "LeniaCPPN"]
    )
    assert get_default_modules("explorers") == {}
    assert get_default_modules("maps") == {
        "MeanBehaviorMap": "adtool_default.maps.MeanBehaviorMap.MeanBehaviorMap",
        "UniformParameterMap": "adtool_default.maps.UniformParameterMap.UniformParameterMap",
        "LeniaStatistics": "adtool_default.maps.LeniaStatistics.LeniaStatistics",
        "UniformParameterMap": "adtool_default.maps.UniformParameterMap.UniformParameterMap",
        "LeniaParameterMap": "adtool_default.maps.LeniaParameterMap.LeniaParameterMap",
        "NEATParameterMap": "adtool_default.maps.NEATParameterMap.NEATParameterMap",
    }
    assert get_default_modules("callbacks") == {}


def test_get_modules():
    assert set(get_modules("systems").keys()) == set(
        ["ExponentialMixture", "Lenia", "LeniaCPPN"]
    )
    assert set(get_modules("explorers").keys()) == set(["IMGEPExplorer"])
    assert set(get_modules("maps").keys()) == set(
        [
            "MeanBehaviorMap",
            "UniformParameterMap",
            "LeniaStatistics",
            "LeniaParameterMap",
            "NEATParameterMap",
        ]
    )
    assert set(get_modules("callbacks").keys()) == set(
        [
            "on_discovery",
            "on_cancelled",
            "on_error",
            "on_finished",
            "on_saved",
            "on_save_finished",
            "interact",
        ]
    )
