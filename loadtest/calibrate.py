"""Калибровка модели времени префилла (§3.3.6.7a).

Зачем. `estimated_ttft` — центральная функция роутера: по ней выбирается
кандидат и принимается решение о раннем отклонении. Пока её коэффициенты
взяты из головы, роутер остаётся эвристикой, а все выводы о SLO —
предположениями. Короткий калибровочный прогон превращает его в
измеримую модель.

Как. Снимаем зависимость «длина входа × длина совпавшего префикса →
время до первого токена» на пустой системе, затем подгоняем

    t_prefill(m) = A + B·m + C·(m/1000)²

где m — число токенов, которые реально надо посчитать (вход минус
попадание в кэш). Квадратичный член обязателен: внимание квадратично по
длине входа, MLP линеен, поэтому линейная аппроксимация врёт именно там,
где это дороже всего — на длинных контекстах.

Подгонка — обычный метод наименьших квадратов на трёх столбцах, без
внешних зависимостей: матрица 3×3 решается явно.

Результат пишется в config/calibration.yaml и подхватывается роутером.
Отсутствие файла — не ошибка: система работает на значениях по умолчанию
и честно помечает модель как неоткалиброванную.

Запуск:
    PYTHONPATH=. uv run python loadtest/calibrate.py --url http://127.0.0.1:9001
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from dataclasses import dataclass

from gateway.router.prefix import BYTES_PER_TOKEN


@dataclass
class Sample:
    prompt_tokens: int
    cached_tokens: int
    compute_tokens: int
    ttft_ms: float


def measure(url: str, prompt: str, *, timeout: float = 120.0) -> tuple[float, dict[str, str]]:
    """Один замер TTFT: время до первого кадра с содержимым."""
    body = json.dumps({
        "model": "mock-llm",
        "stream": True,
        "mock_output_tokens": 2,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        headers = dict(resp.headers)
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            if '"content"' in line:
                return (time.perf_counter() - start) * 1000.0, headers
    raise RuntimeError("апстрим не отдал ни одного токена с содержимым")


def collect(url: str, *, repeats: int = 3, salt: str | None = None) -> list[Sample]:
    """Снимает точки по длине входа и по доле попадания в кэш.

    Соль обязательна: KV-кэш апстрима переживает прогон, и повтор тех же
    промптов измерял бы тёплый кэш вместо холодного. На этой ошибке уже
    один раз получилось, что прокси «быстрее» прямого обращения.
    """
    salt = salt or f"cal{int(time.time())}"
    samples: list[Sample] = []

    lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    for n in lengths:
        prompt = f"|{salt}-{n}|" + "a" * (n * BYTES_PER_TOKEN)
        ttfts = []
        headers = {}
        for r in range(repeats):
            # Каждый повтор — свой промпт, иначе второй пойдёт по тёплому кэшу.
            p = f"|{salt}-{n}-{r}|" + prompt[len(f"|{salt}-{n}|"):]
            ms, headers = measure(url, p)
            ttfts.append(ms)
        prompt_tokens = int(headers.get("X-Mock-Prompt-Tokens", n))
        cached = int(headers.get("X-Mock-Cached-Tokens", 0))
        samples.append(Sample(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            compute_tokens=prompt_tokens - cached,
            ttft_ms=statistics.median(ttfts),
        ))
        print(f"  вход {prompt_tokens:>6} токенов → TTFT {statistics.median(ttfts):>8.1f} мс "
              f"(разброс {min(ttfts):.1f}…{max(ttfts):.1f})")

    # Точки с попаданием в кэш: тот же вход второй раз.
    print("  — точки с попаданием в кэш —")
    for n in (4096, 16384):
        p = f"|{salt}-warm-{n}|" + "b" * (n * BYTES_PER_TOKEN)
        measure(url, p)                      # прогрев
        ms, headers = measure(url, p)        # замер по тёплому
        prompt_tokens = int(headers.get("X-Mock-Prompt-Tokens", n))
        cached = int(headers.get("X-Mock-Cached-Tokens", 0))
        samples.append(Sample(prompt_tokens, cached, prompt_tokens - cached, ms))
        print(f"  вход {prompt_tokens:>6}, из кэша {cached:>6} → считать "
              f"{prompt_tokens - cached:>6} → TTFT {ms:>8.1f} мс")

    return samples


def fit(samples: list[Sample]) -> dict[str, float]:
    """МНК для t = A + B·m + C·k², где k = m/1000.

    Решается нормальная система 3×3 методом Гаусса. Внешние зависимости
    здесь были бы дороже, чем двадцать строк арифметики.
    """
    rows = [(1.0, float(s.compute_tokens), (s.compute_tokens / 1000.0) ** 2, s.ttft_ms)
            for s in samples]
    n = 3
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for *x, y in rows:
        for i in range(n):
            atb[i] += x[i] * y
            for j in range(n):
                ata[i][j] += x[i] * x[j]

    # Гаусс с выбором ведущего элемента.
    m = [ata[i] + [atb[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("вырожденная система: точки лежат слишком близко")
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    coeffs = [m[i][n] / m[i][i] for i in range(n)]
    return {"const_ms": coeffs[0], "per_token_ms": coeffs[1], "quadratic_ms": coeffs[2]}


def residuals(samples: list[Sample], c: dict[str, float]) -> dict[str, float]:
    """Качество подгонки. Без него коэффициенты — просто числа."""
    errs = []
    for s in samples:
        k = s.compute_tokens / 1000.0
        pred = c["const_ms"] + c["per_token_ms"] * s.compute_tokens + c["quadratic_ms"] * k * k
        errs.append(abs(pred - s.ttft_ms))
    rel = [e / s.ttft_ms for e, s in zip(errs, samples) if s.ttft_ms > 0]
    return {
        "mae_ms": statistics.fmean(errs),
        "max_err_ms": max(errs),
        "mean_rel_err": statistics.fmean(rel) if rel else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:9001", help="адрес апстрима")
    ap.add_argument("--repeats", type=int, default=3, help="замеров на точку (берётся медиана)")
    ap.add_argument("--out", default="config/calibration.yaml")
    args = ap.parse_args()

    print(f"Калибровка модели префилла по {args.url}")
    print("Прогон обязан идти на пустой системе: конкуренция исказит замер.\n")

    samples = collect(args.url, repeats=args.repeats)
    coeffs = fit(samples)
    q = residuals(samples, coeffs)

    print(f"\n  t_prefill(m) = {coeffs['const_ms']:.2f} "
          f"+ {coeffs['per_token_ms']:.5f}·m "
          f"+ {coeffs['quadratic_ms']:.3f}·(m/1000)²   [мс]")
    print(f"  средняя ошибка {q['mae_ms']:.1f} мс, максимальная {q['max_err_ms']:.1f} мс, "
          f"относительная {q['mean_rel_err']:.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Коэффициенты модели времени префилла (§3.3.6.7a).\n")
        f.write("# Файл создаётся loadtest/calibrate.py. Правка руками бессмысленна:\n")
        f.write("# значения имеют смысл только как результат измерения на конкретном стенде.\n")
        f.write(f"# Снято: {time.strftime('%Y-%m-%d %H:%M:%S')}, апстрим {args.url}\n")
        f.write(f"# Средняя ошибка {q['mae_ms']:.1f} мс ({q['mean_rel_err']:.1%}).\n\n")
        f.write("prefill:\n")
        for k, v in coeffs.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\nquality:\n")
        for k, v in q.items():
            f.write(f"  {k}: {v:.6f}\n")
    print(f"\n  записано в {args.out}")


if __name__ == "__main__":
    main()
