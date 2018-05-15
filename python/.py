# python3


def range2(dot1, dot2):
    return (dot1[0] - dot2[0]) * (dot1[0] - dot2[0]) + (dot1[1] - dot2[1]) * (dot1[1] - dot2[1])


def range_x(dot1, dot2):
    return abs(dot1[0] - dot2[0])


def range_y(dot1, dot2):
    return abs(dot1[1] - dot2[1])


def merge(ar, lf, mid, r, midx, d):
    lf1 = lf
    l1 = mid - lf
    l2 = r - mid
    tmp = [0 for _ in range(l1 + l2)]
    i = 0
    stripe = list()
    while l1 > 0 and l2 > 0:
        if ar[lf][1] > ar[mid][1]:
            tmp[i] = ar[mid]
            if abs(ar[mid][0] - midx) < d:
                stripe.append(ar[mid])
            mid += 1
            l2 -= 1
        else:
            tmp[i] = ar[lf]
            if abs(ar[lf][0] - midx) < d:
                stripe.append(ar[lf])
            lf += 1
            l1 -= 1
        i += 1
    while l1 > 0:
        tmp[i] = ar[lf]
        if abs(ar[lf][0] - midx) < d:
            stripe.append(ar[lf])
        lf += 1
        i += 1
        l1 -= 1
    while l2 > 0:
        tmp[i] = ar[mid]
        if abs(ar[mid][0] - midx) < d:
            stripe.append(ar[mid])
        mid += 1
        i += 1
        l2 -= 1
    i = 0
    for j in range(lf1, r):
        ar[j] = tmp[i]
        i += 1
    return tuple(stripe)


def min_range(d_s_x, lf, r):
    if r - lf == 2:
        if d_s_x[lf][1] > d_s_x[lf + 1][1]:
            d_s_x[lf], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf]
        return range2(d_s_x[lf], d_s_x[r - 1])
    elif r - lf == 3:
        """
        for i_i in reversed(range(2)):
            for j_j in range(i_i):
                if d_s_x[j_j] > d_s_x[j_j + 1]:
                    d_s_x[j_j], d_s_x[j_j + 1] = d_s_x[j_j + 1], d_s_x[j_j]
        """
        if d_s_x[lf] > d_s_x[lf + 1]:
            d_s_x[lf], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf]
            if d_s_x[lf + 1] > d_s_x[lf + 2]:
                d_s_x[lf + 2], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf + 2]
                if d_s_x[lf] > d_s_x[lf + 1]:
                    d_s_x[lf], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf]
        else:
            if d_s_x[lf + 1] > d_s_x[lf + 2]:
                d_s_x[lf + 2], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf + 2]
                if d_s_x[lf] > d_s_x[lf + 1]:
                    d_s_x[lf], d_s_x[lf + 1] = d_s_x[lf + 1], d_s_x[lf]
        return min(
                   range2(d_s_x[lf], d_s_x[r - 2]),
                   range2(d_s_x[lf + 1], d_s_x[r - 1]),
                   range2(d_s_x[lf], d_s_x[r - 1])
                  )
    mid = (lf + r) // 2
    midx = d_s_x[mid][0]
    d1 = min_range(d_s_x, lf, mid)
    d2 = min_range(d_s_x, mid, r)
    d = min(d1, d2)
    stripe = merge(d_s_x, lf, mid, r, midx, d)
    if len(stripe) < 2:
        return d
    d_s = range2(stripe[0], stripe[1])
    for i in range(len(stripe) - 1):
        j = i + 1
        while j < i + 7:
            d_s = min(d_s, range2(stripe[i], stripe[j]))
            j += 1
            if j == len(stripe):
                break
    return min(d_s, d)


def first(el):
    return el[0]


def second(el):
    return el[1]


def main_main():
    n = int(input())
    dots_sorted_x = []
    for _ in range(n):
        dots_sorted_x.append(tuple(map(int, input().split())))
    dots_sorted_x.sort(key=first)
    for i in range(len(dots_sorted_x) - 1):
        if dots_sorted_x[i] == dots_sorted_x[i + 1]:
            print(0.0)
            exit()
    res = float("{0:.4f}".format(pow(min_range(dots_sorted_x, 0, len(dots_sorted_x)), 0.5)))
    print(res)


main_main()