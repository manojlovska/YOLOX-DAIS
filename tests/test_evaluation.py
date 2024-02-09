def test_val_dataset2():
    from train_yolino import Exp
    print("yey")

    exp = Exp()
    val_dataset = exp.get_eval_dataset()
    import pdb
    pdb.set_trace()
    assert val_dataset is None


def test_val_dataset():
    from train_custom_data import Exp

    exp = Exp()
    val_dataset = exp.get_eval_dataset()
    import pdb
    pdb.set_trace()
    assert val_dataset is None

def test_evaluator():
    from train_custom_data import Exp

    exp = Exp()
    evaluator = exp.get_evaluator(16, False, testdev=False, legacy=False)
    import pdb
    pdb.set_trace()
    assert evaluator is None