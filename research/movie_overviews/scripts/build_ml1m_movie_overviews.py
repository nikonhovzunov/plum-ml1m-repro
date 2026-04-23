from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from requests import HTTPError


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOVIES_PATH = ROOT / "data" / "raw" / "ml-1m" / "movies.dat"
DEFAULT_OUTPUT_PATH = ROOT / "research" / "movie_overviews" / "data" / "ml1m_movie_overviews.csv"
DEFAULT_CACHE_PATH = ROOT / "research" / "movie_overviews" / "cache" / "movie_overviews_cache.jsonl"
DEFAULT_DESCRIPTOR_CACHE_PATH = (
    ROOT / "research" / "movie_overviews" / "cache" / "hf_movie_descriptors.parquet"
)
DEFAULT_DESCRIPTOR_URL = (
    "https://huggingface.co/datasets/mt0rm0/movie_descriptors/resolve/main/descriptors_data.parquet"
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
TMDB_API = "https://api.themoviedb.org/3/search/movie"
OMDB_API = "https://www.omdbapi.com/"

USER_AGENT = (
    "plum-ml1m-repro/0.1 "
    "(MovieLens-1M movie overview enrichment; contact: local-research-script)"
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
PARENS_RE = re.compile(r"\([^)]*\)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACES_RE = re.compile(r"\s+")

BAD_PAGE_HINTS = (
    "disambiguation",
    "soundtrack",
    "album",
    "song",
    "novel",
    "book",
    "video game",
    "television series",
    "tv series",
    "miniseries",
    "musical",
    "play",
    "opera",
)

FILM_HINTS = (
    " film",
    " movie",
    " documentary",
    " animated ",
    " cinema",
    " motion picture",
)


@dataclass(frozen=True)
class MovieRow:
    movie_id: int
    title: str
    clean_title: str
    search_title: str
    year: int | None
    genres: str


@dataclass(frozen=True)
class MatchResult:
    overview: str
    source: str
    status: str
    provider: str = ""
    matched_title: str = ""
    score: float | None = None
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich MovieLens-1M movies with conservative external overview matches."
    )
    parser.add_argument("--movies-path", type=Path, default=DEFAULT_MOVIES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N limit for smoke runs.")
    parser.add_argument("--sleep", type=float, default=0.03, help="Sleep between uncached external lookups.")
    parser.add_argument("--force", action="store_true", help="Ignore cached records and refetch.")
    parser.add_argument(
        "--providers",
        default="hf,tmdb,omdb",
        help=(
            "Comma-separated provider order. 'hf' uses the CC0 Hugging Face movie_descriptors "
            "dataset derived from Kaggle/TMDb. TMDb/OMDb are used only when API keys are available."
        ),
    )
    parser.add_argument("--descriptor-url", default=DEFAULT_DESCRIPTOR_URL)
    parser.add_argument("--descriptor-cache-path", type=Path, default=DEFAULT_DESCRIPTOR_CACHE_PATH)
    parser.add_argument("--wikipedia-candidates", type=int, default=5)
    return parser.parse_args()


def move_trailing_article(title: str) -> str:
    articles = "The|A|An|Le|La|Les|Il|El|Los|Las|Der|Die|Das|L'"
    match = re.match(rf"^(.*),\s+({articles})$", title, flags=re.IGNORECASE)
    if not match:
        return title
    return f"{match.group(2)} {match.group(1)}"


def extract_title_year(raw_title: str) -> tuple[str, str, int | None]:
    year: int | None = None
    title = raw_title.strip()
    match = YEAR_RE.search(title)
    if match:
        year = int(match.group(1))
        title = YEAR_RE.sub("", title).strip()
    return title, move_trailing_article(title), year


def normalize_title(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = PARENS_RE.sub(" ", value)
    value = re.sub(r"^(the|a|an)\s+", "", value)
    value = NON_ALNUM_RE.sub(" ", value)
    return SPACES_RE.sub(" ", value).strip()


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = SPACES_RE.sub(" ", value.strip())
        if not value:
            continue
        key = normalize_title(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def title_variants(movie: MovieRow) -> list[str]:
    variants = [movie.search_title, movie.clean_title]
    no_parentheses = PARENS_RE.sub(" ", movie.search_title)
    variants.append(no_parentheses)

    parenthetical_aliases = re.findall(r"\(([^)]{2,80})\)", movie.clean_title)
    for alias in parenthetical_aliases:
        if not re.fullmatch(r"\d{4}", alias):
            variants.append(alias)

    before_parentheses = PARENS_RE.sub("", movie.clean_title).strip()
    variants.append(move_trailing_article(before_parentheses))

    if ":" in movie.search_title:
        before_colon, after_colon = movie.search_title.split(":", 1)
        variants.extend([before_colon, after_colon])

    if " - " in movie.search_title:
        before_dash, after_dash = movie.search_title.split(" - ", 1)
        variants.extend([before_dash, after_dash])

    return dedupe_keep_order(variants)


def similarity(a: str, b: str) -> float:
    norm_a = normalize_title(a)
    norm_b = normalize_title(b)
    if not norm_a or not norm_b:
        return 0.0

    seq = SequenceMatcher(None, norm_a, norm_b).ratio()
    toks_a = set(norm_a.split())
    toks_b = set(norm_b.split())
    overlap = len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)
    containment = min(
        len(toks_a & toks_b) / max(len(toks_a), 1),
        len(toks_a & toks_b) / max(len(toks_b), 1),
    )
    return max(seq, overlap, containment)


def read_movies(path: Path) -> list[MovieRow]:
    rows: list[MovieRow] = []
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    for row in movies.itertuples(index=False):
        clean_title, search_title, year = extract_title_year(str(row.title))
        rows.append(
            MovieRow(
                movie_id=int(row.movie_id),
                title=str(row.title),
                clean_title=clean_title,
                search_title=search_title,
                year=year,
                genres=str(row.genres),
            )
        )
    return rows


def load_descriptors(url: str, cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        descriptors = pd.read_parquet(cache_path)
    else:
        descriptors = pd.read_parquet(url)
        descriptors.to_parquet(cache_path, index=False)

    required = {"title", "release_year", "overview"}
    missing = required - set(descriptors.columns)
    if missing:
        raise ValueError(f"Descriptor dataset is missing columns: {sorted(missing)}")

    descriptors = descriptors.dropna(subset=["title", "release_year", "overview"]).copy()
    descriptors["release_year"] = descriptors["release_year"].astype(int)
    descriptors["overview"] = descriptors["overview"].astype(str).str.strip()
    descriptors["norm_title"] = descriptors["title"].map(normalize_title)
    descriptors = descriptors[descriptors["overview"].map(overview_is_valid)]
    return descriptors.reset_index(drop=True)


def load_cache(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "movie_id" in record:
                cache[int(record["movie_id"])] = record
    return cache


def append_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int = 20,
    max_retries: int = 6,
) -> dict[str, Any]:
    last_error: HTTPError | None = None
    for attempt in range(max_retries + 1):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code not in TRANSIENT_STATUS_CODES:
            response.raise_for_status()
            return response.json()

        try:
            response.raise_for_status()
        except HTTPError as exc:
            last_error = exc

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            sleep_seconds = min(float(retry_after), 90.0)
        else:
            sleep_seconds = min(2.0 * (2**attempt), 90.0)
        print(
            f"Transient HTTP {response.status_code}; retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s",
            flush=True,
        )
        time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    response.raise_for_status()
    return response.json()


def overview_is_valid(text: str) -> bool:
    cleaned = SPACES_RE.sub(" ", (text or "").strip())
    if len(cleaned) < 40:
        return False
    if cleaned.lower() in {"n/a", "none", "null"}:
        return False
    return True


def try_tmdb(session: requests.Session, movie: MovieRow) -> MatchResult | None:
    api_key = os.environ.get("TMDB_API_KEY")
    bearer = os.environ.get("TMDB_BEARER_TOKEN")
    if not api_key and not bearer:
        return None

    headers = {"User-Agent": USER_AGENT}
    params: dict[str, Any] = {"query": movie.search_title, "include_adult": "false", "language": "en-US"}
    if movie.year:
        params["year"] = movie.year
    if api_key:
        params["api_key"] = api_key
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    data = session.get(TMDB_API, params=params, headers=headers, timeout=20)
    data.raise_for_status()
    results = data.json().get("results", [])[:5]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for item in results:
        title = item.get("title") or item.get("original_title") or ""
        title_score = similarity(movie.search_title, title)
        release_year = None
        if item.get("release_date"):
            release_year = int(str(item["release_date"])[:4])
        year_match = movie.year is None or release_year == movie.year
        overview = item.get("overview") or ""
        if not overview_is_valid(overview):
            continue
        score = title_score + (0.25 if year_match else -0.2)
        candidates.append((score, item))

    if not candidates:
        return MatchResult("", "", "no_description", provider="tmdb", reason="no_tmdb_candidate")

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else -1.0
    if best_score < 1.05 or second_score >= best_score - 0.04:
        return MatchResult("", "", "no_description", provider="tmdb", reason="ambiguous_tmdb", score=best_score)

    source = f"https://www.themoviedb.org/movie/{best.get('id')}"
    return MatchResult(
        overview=SPACES_RE.sub(" ", best.get("overview", "").strip()),
        source=source,
        status="found",
        provider="tmdb",
        matched_title=best.get("title") or "",
        score=best_score,
    )


def try_omdb(session: requests.Session, movie: MovieRow) -> MatchResult | None:
    api_key = os.environ.get("OMDB_API_KEY")
    if not api_key:
        return None

    params: dict[str, Any] = {"apikey": api_key, "t": movie.search_title, "plot": "short", "r": "json"}
    if movie.year:
        params["y"] = movie.year
    data = request_json(session, OMDB_API, params=params)
    if data.get("Response") != "True":
        return MatchResult("", "", "no_description", provider="omdb", reason=data.get("Error", "not_found"))

    title_score = similarity(movie.search_title, data.get("Title", ""))
    year_text = str(data.get("Year", ""))
    year_match = movie.year is None or str(movie.year) in year_text
    plot = data.get("Plot", "")
    if title_score < 0.88 or not year_match or not overview_is_valid(plot):
        return MatchResult("", "", "no_description", provider="omdb", reason="low_confidence_omdb", score=title_score)

    imdb_id = data.get("imdbID", "")
    source = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else "omdb"
    return MatchResult(
        overview=SPACES_RE.sub(" ", plot.strip()),
        source=source,
        status="found",
        provider="omdb",
        matched_title=data.get("Title", ""),
        score=title_score + 0.25,
    )


def try_hf_descriptors(movie: MovieRow, descriptors: pd.DataFrame | None) -> MatchResult | None:
    if descriptors is None:
        return None

    variants = title_variants(movie)
    if not variants:
        return MatchResult("", "", "no_description", provider="hf_movie_descriptors", reason="empty_normalized_title")

    candidates = descriptors
    if movie.year is not None:
        candidates = candidates[candidates["release_year"].eq(int(movie.year))]
    if candidates.empty:
        return MatchResult("", "", "no_description", provider="hf_movie_descriptors", reason="no_year_candidates")

    direct_frames: list[pd.DataFrame] = []
    for priority, variant in enumerate(variants):
        norm_query = normalize_title(variant)
        matched = candidates[candidates["norm_title"].eq(norm_query)].copy()
        if not matched.empty:
            matched["score"] = 1.0 - (priority * 0.01)
            matched["matched_variant"] = variant
            direct_frames.append(matched)

    direct = pd.concat(direct_frames, ignore_index=True) if direct_frames else pd.DataFrame()
    if direct.empty:
        # Conservative fuzzy fallback: keep only very close title matches in the same release year.
        sample = candidates[["title", "norm_title", "overview"]].copy()
        sample["score"] = sample["title"].map(
            lambda title: max(similarity(variant, str(title)) for variant in variants[:3])
        )
        sample["matched_variant"] = variants[0]
        direct = sample[sample["score"].ge(0.96)].sort_values("score", ascending=False)

    if direct.empty:
        return MatchResult("", "", "no_description", provider="hf_movie_descriptors", reason="no_title_match")

    direct = direct.sort_values(["score", "title"], ascending=[False, True])
    best = direct.iloc[0]
    if len(direct) > 1:
        second = direct.iloc[1]
        if float(second["score"]) >= float(best["score"]) - 0.01 and str(second["overview"]) != str(best["overview"]):
            return MatchResult(
                "",
                "",
                "no_description",
                provider="hf_movie_descriptors",
                matched_title=str(best["title"]),
                score=float(best["score"]),
                reason="ambiguous_hf_match",
            )

    return MatchResult(
        overview=SPACES_RE.sub(" ", str(best["overview"]).strip()),
        source="https://huggingface.co/datasets/mt0rm0/movie_descriptors",
        status="found",
        provider="hf_movie_descriptors",
        matched_title=str(best["title"]),
        score=float(best["score"]),
        reason="strict_title_year_match",
    )


def wikipedia_search(session: requests.Session, query: str, limit: int) -> list[dict[str, Any]]:
    data = request_json(
        session,
        WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1,
        },
    )
    return data.get("query", {}).get("search", [])


def wikipedia_extracts(session: requests.Session, titles: list[str]) -> dict[str, dict[str, str]]:
    if not titles:
        return {}
    data = request_json(
        session,
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "titles": "|".join(titles[:10]),
            "format": "json",
            "utf8": 1,
        },
    )
    pages = data.get("query", {}).get("pages", {})
    return {
        page.get("title", ""): {
            "title": page.get("title", ""),
            "extract": page.get("extract", ""),
        }
        for page in pages.values()
        if "missing" not in page and page.get("title")
    }


def page_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def score_wikipedia_candidate(movie: MovieRow, title: str, snippet: str, extract: str) -> tuple[float, str]:
    combined = f"{title} {snippet} {extract[:800]}".lower()
    title_lower = title.lower()
    first_sentence = re.split(r"(?<=[.!?])\s+", extract.strip(), maxsplit=1)[0].lower()
    title_score = similarity(movie.search_title, title)
    year_match = movie.year is not None and str(movie.year) in combined
    film_hint = any(hint in combined for hint in FILM_HINTS) or "(film)" in title_lower
    first_sentence_film = any(hint in first_sentence for hint in FILM_HINTS) or "(film)" in title_lower
    first_sentence_year_film = (
        movie.year is not None
        and str(movie.year) in first_sentence
        and first_sentence_film
    )
    is_disambiguation = "disambiguation" in title.lower() or "may refer to:" in combined[:400]
    nonfilm_title_hint = any(hint in title_lower for hint in BAD_PAGE_HINTS)
    nonfilm_first_sentence = any(hint in first_sentence for hint in BAD_PAGE_HINTS) and not first_sentence_film
    franchise_hint = "franchise" in first_sentence and not first_sentence_year_film
    bad_hint = is_disambiguation or nonfilm_title_hint or nonfilm_first_sentence or franchise_hint

    score = title_score
    if film_hint:
        score += 0.25
    if first_sentence_film:
        score += 0.12
    if first_sentence_year_film:
        score += 0.25
    if normalize_title(movie.search_title) == normalize_title(title):
        score += 0.05
    if "(film)" in title_lower:
        score += 0.08
    if year_match:
        score += 0.25
    elif movie.year is not None:
        score -= 0.12
    if bad_hint:
        score -= 0.4
    if is_disambiguation:
        score -= 0.8

    reason = (
        f"title_score={title_score:.3f};film={film_hint};first_film={first_sentence_film};"
        f"first_year_film={first_sentence_year_film};year={year_match};bad={bad_hint}"
    )
    return score, reason


def select_wikipedia_candidate(
    movie: MovieRow,
    candidates: dict[str, dict[str, str]],
    min_score: float = 1.08,
) -> MatchResult:
    scored: list[tuple[float, str, str, dict[str, Any]]] = []
    for title, candidate in candidates.items():
        extract = candidate.get("extract", "")
        score, reason = score_wikipedia_candidate(
            movie=movie,
            title=title,
            snippet=candidate.get("snippet", ""),
            extract=extract,
        )
        if overview_is_valid(extract):
            scored.append((score, title, reason, {"extract": extract}))

    if not scored:
        return MatchResult("", "", "no_description", provider="wikipedia", reason="no_wikipedia_candidate")

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_title, reason, payload = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0

    best_extract = payload["extract"].lower()
    best_title_lower = best_title.lower()
    best_first_sentence = re.split(r"(?<=[.!?])\s+", payload["extract"].strip(), maxsplit=1)[0].lower()
    best_has_film_hint = any(hint in f"{best_title_lower} {best_first_sentence}" for hint in FILM_HINTS) or "(film)" in best_title_lower
    if "disambiguation" in best_title_lower or "may refer to:" in best_extract[:400]:
        return MatchResult(
            "", "", "no_description", provider="wikipedia", matched_title=best_title, score=best_score, reason=f"disambiguation;{reason}"
        )
    if not best_has_film_hint:
        return MatchResult(
            "", "", "no_description", provider="wikipedia", matched_title=best_title, score=best_score, reason=f"non_film_best;{reason}"
        )

    if best_score < min_score:
        return MatchResult(
            "", "", "no_description", provider="wikipedia", matched_title=best_title, score=best_score, reason=f"low_confidence;{reason}"
        )

    if second_score >= best_score - 0.04:
        return MatchResult(
            "", "", "no_description", provider="wikipedia", matched_title=best_title, score=best_score, reason=f"ambiguous;{reason}"
        )

    overview = SPACES_RE.sub(" ", payload["extract"].strip())
    return MatchResult(
        overview=overview,
        source=page_url(best_title),
        status="found",
        provider="wikipedia",
        matched_title=best_title,
        score=best_score,
        reason=reason,
    )


def direct_wikipedia_titles(movie: MovieRow) -> list[str]:
    titles = [movie.search_title]
    if movie.year:
        titles.extend(
            [
                f"{movie.search_title} ({movie.year} film)",
                f"{movie.search_title} (film)",
            ]
        )
    else:
        titles.append(f"{movie.search_title} (film)")
    seen: set[str] = set()
    unique = []
    for title in titles:
        if title not in seen:
            seen.add(title)
            unique.append(title)
    return unique


def try_wikipedia(session: requests.Session, movie: MovieRow, candidate_limit: int) -> MatchResult:
    direct = wikipedia_extracts(session, direct_wikipedia_titles(movie))
    if direct:
        direct_result = select_wikipedia_candidate(movie, direct, min_score=1.08)
        if direct_result.status == "found":
            return direct_result

    queries = []
    if movie.year:
        queries.append(f'"{movie.search_title}" {movie.year} film')
        queries.append(f'{movie.search_title} {movie.year} film')
    queries.append(f'"{movie.search_title}" film')
    queries.append(f'{movie.search_title} film')

    raw_candidates: dict[str, dict[str, Any]] = {}
    for query in queries:
        for result in wikipedia_search(session, query, candidate_limit):
            title = result.get("title", "")
            if title and title not in raw_candidates:
                raw_candidates[title] = result

    extracts = wikipedia_extracts(session, list(raw_candidates))
    merged_candidates: dict[str, dict[str, str]] = {}
    for title, result in raw_candidates.items():
        extract_title = extracts.get(title, {}).get("title", title)
        merged_candidates[extract_title] = {
            "title": extract_title,
            "extract": extracts.get(title, {}).get("extract", ""),
            "snippet": result.get("snippet", ""),
        }
    return select_wikipedia_candidate(movie, merged_candidates, min_score=1.08)


def lookup_movie(
    session: requests.Session,
    movie: MovieRow,
    providers: list[str],
    wikipedia_candidates: int,
    descriptors: pd.DataFrame | None = None,
) -> MatchResult:
    for provider in providers:
        try:
            if provider == "hf":
                result = try_hf_descriptors(movie, descriptors)
            elif provider == "tmdb":
                result = try_tmdb(session, movie)
            elif provider == "omdb":
                result = try_omdb(session, movie)
            elif provider == "wikipedia":
                result = try_wikipedia(session, movie, wikipedia_candidates)
            else:
                continue
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in TRANSIENT_STATUS_CODES:
                raise
            result = MatchResult("", "", "no_description", provider=provider, reason=f"error:{type(exc).__name__}:{exc}")
        except Exception as exc:
            result = MatchResult("", "", "no_description", provider=provider, reason=f"error:{type(exc).__name__}:{exc}")

        if result is None:
            continue
        if result.status == "found":
            return result

    return MatchResult("", "", "no_description", provider="", reason="all_providers_failed_or_unavailable")


def cache_record(movie: MovieRow, result: MatchResult) -> dict[str, Any]:
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genres,
        "overview": result.overview,
        "source": result.source,
        "status": result.status,
        "provider": result.provider,
        "matched_title": result.matched_title,
        "score": result.score,
        "reason": result.reason,
    }


def output_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "movie_id": int(record["movie_id"]),
        "title": record["title"],
        "year": "" if record.get("year") is None else int(record["year"]),
        "genres": record["genres"],
        "overview": record.get("overview", "") if record.get("status") == "found" else "",
        "source": record.get("source", "") if record.get("status") == "found" else "",
        "status": "found" if record.get("status") == "found" else "no_description",
    }


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    movies = read_movies(args.movies_path)
    if args.limit is not None:
        movies = movies[: args.limit]

    cache = {} if args.force else load_cache(args.cache_path)
    providers = [provider.strip().lower() for provider in args.providers.split(",") if provider.strip()]
    descriptors = load_descriptors(args.descriptor_url, args.descriptor_cache_path) if "hf" in providers else None
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    records: list[dict[str, Any]] = []
    found = 0
    for idx, movie in enumerate(movies, start=1):
        if movie.movie_id in cache:
            record = cache[movie.movie_id]
        else:
            result = lookup_movie(
                session,
                movie,
                providers=providers,
                wikipedia_candidates=args.wikipedia_candidates,
                descriptors=descriptors,
            )
            record = cache_record(movie, result)
            append_cache(args.cache_path, record)
            if args.sleep:
                time.sleep(args.sleep)

        if record.get("status") == "found":
            found += 1
        records.append(output_record(record))

        if idx == 1 or idx % 100 == 0 or idx == len(movies):
            rate = found / idx
            print(f"[{idx:4d}/{len(movies)}] found={found:4d} coverage={rate:.1%}", flush=True)

    df = pd.DataFrame(records, columns=["movie_id", "title", "year", "genres", "overview", "source", "status"])
    df.to_csv(args.output_path, index=False, encoding="utf-8")
    return df


def main() -> None:
    args = parse_args()
    started = time.time()
    df = build_dataset(args)
    summary = {
        "rows": int(len(df)),
        "found": int((df["status"] == "found").sum()),
        "no_description": int((df["status"] == "no_description").sum()),
        "coverage": float((df["status"] == "found").mean()) if len(df) else 0.0,
        "output_path": str(args.output_path),
        "cache_path": str(args.cache_path),
        "seconds": round(time.time() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
