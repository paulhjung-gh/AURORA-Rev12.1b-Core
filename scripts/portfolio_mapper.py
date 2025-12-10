def map_portfolio(target_weights, isa_monthly=1650000, direct_monthly=2350000):
    """
    target_weights = {
        "spx": 0.36,
        "ndx": 0.17,
        "div": 0.16,
        "em": 0.02,
        "energy": 0.01,
        "duration": 0.05,
        "sgov": 0.22,
        "gold": 0.05
    }
    """

    total = isa_monthly + direct_monthly

    isa_allowed = ["spx", "ndx", "div", "gold"]
    direct_only = ["em", "energy", "duration", "sgov"]

    isa_alloc = {}
    direct_alloc = {}

    for k, v in target_weights.items():
        amt = total * v
        if k in isa_allowed:
            isa_alloc[k] = min(amt, isa_monthly)  # 기본 매핑
        else:
            isa_alloc[k] = 0

    direct_alloc = {
        k: total * target_weights[k] - isa_alloc.get(k, 0)
        for k in target_weights
    }

    return isa_alloc, direct_alloc

if __name__ == "__main__":
    t = {
        "spx": 0.36,
        "ndx": 0.17,
        "div": 0.16,
        "em": 0.02,
        "energy": 0.01,
        "duration": 0.0,
        "sgov": 0.22,
        "gold": 0.05
    }

    ia, da = map_portfolio(t)
    print("ISA:", ia)
    print("DIRECT:", da)
