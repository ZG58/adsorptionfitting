def langmuir(P, q_max, b):
    """
    Langmuir 等温线模型
    计算公式为 (q_max * b * P) / (1 + b * P)

    参数:
    P : 浓度或压力值
    q_max : 最大吸附量，表示在表面上可以吸附的最大分子数量
    b : Langmuir 常数，表示吸附平衡常数，影响吸附能力
    """
    return (q_max * b * P) / (1 + b * P)


def toth(P, q_max, b, n):
    """
    Toth 等温线模型
    计算公式为 (q_max * b * P) / ((1 + (b * P)^(1/n))^n)

    参数:
    P : 浓度或压力值
    q_max : 最大吸附量，表示在表面上可以吸附的最大分子数量
    b : Toth 常数，表示吸附平衡常数，影响吸附能力
    n : 异质性指数，影响吸附表面的均匀性
    """
    return (q_max * b * P) / ((1 + (b * P) ** (1 / n)) ** n)


def freundlich(P, k, n):
    """
    Freundlich 等温线模型
    计算公式为 k * P^(1/n)

    参数:
    P : 浓度或压力值
    k : Freundlich 常数，表示吸附能力
    n : 异质性指数，影响吸附表面的均匀性
    """
    return k * P ** (1 / n)


def temkin(P, A, B):
    """
    Temkin 等温线模型
    计算公式为 A * ln(B * P)

    参数:
    P : 浓度或压力值
    A : Temkin 常数
    B : Temkin 常数
    """
    import numpy as np
    return A * np.log(B * P)


def dubinin_radushkevich(P, q_max, beta):
    """
    Dubinin-Radushkevich 等温线模型
    计算公式为 q_max * exp(-beta * (ln(P))^2)

    参数:
    P : 浓度或压力值
    q_max : 最大吸附量
    beta : Dubinin 常数
    """
    import numpy as np
    return q_max * np.exp(-beta * (np.log(P)) ** 2)


def dual_site(P, q_max, b1, b2, n1, n2):
    """
    Dual-site 等温线模型
    计算公式为 (q_max * b1 * P) / (1 + b1 * P) + (q_max * b2 * P) / (1 + b2 * P)

    参数:
    P : 浓度或压力值
    q_max : 最大吸附量
    b1 : 第一位点的吸附常数
    b2 : 第二位点的吸附常数
    n1 : 第一位点的异质性指数
    n2 : 第二位点的异质性指数
    """
    return (q_max * b1 * P) / (1 + b1 * P) + (q_max * b2 * P) / (1 + b2 * P)


def sips(P, q_max, b, n):
    """
    Sips 等温线模型
    计算公式为 (q_max * b * P^n) / (1 + b * P^n)

    参数:
    P : 浓度或压力值
    q_max : 最大吸附量
    b : Sips 常数
    n : Sips 异质性指数
    """
    return (q_max * b * P ** n) / (1 + b * P ** n)


def get_adsorption_models():
    """返回可用的吸附模型列表。"""
    return [
        'Langmuir',
        'Freundlich',
        'BET',
        'Temkin',
        'Dubinin-Radushkevich',
        'Toth',
        'Dual-site',
        'Sips'
    ]


if __name__ == "__main__":
    # 测试各个模型
    P = 1.0  # 示例浓度或压力值
    q_max = 10.0  # 示例最大吸附量
    b = 0.5  # 示例常数
    n = 2.0  # 示例指数
    print("Langmuir:", langmuir(P, q_max, b))
    print("Toth:", toth(P, q_max, b, n))
    print("Freundlich:", freundlich(P, 1.0, n))
    print("Temkin:", temkin(P, 1.0, 1.0))
    print("Dubinin-Radushkevich:", dubinin_radushkevich(P, q_max, b))
    print("Dual-site:", dual_site(P, q_max, b, b, n, n))
    print("Sips:", sips(P, q_max, b, n))
