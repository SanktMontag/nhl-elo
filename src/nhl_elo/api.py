"""Small cached client for the (unofficial, undocumented) NHL APIs.

Two hosts are used:
  api-web.nhle.com/v1/      season list, standings (divisions)
  api.nhle.com/stats/rest/  every game of a season in one call, team and franchise tables

Community reference: https://github.com/Zmalski/NHL-API-Reference
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import CACHE_DIR, Settings, load_settings

WEB = "https://api-web.nhle.com/v1/"
STATS = "https://api.nhle.com/stats/rest/en/"


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(self, settings: Settings | None = None, cache_dir: Path = CACHE_DIR, offline: bool = False):
        self.settings = settings or load_settings()
        self.cache_dir = cache_dir
        self.offline = offline
        self._last = 0.0
        self.requests = 0

    def get(self, url: str, cache: str, max_age_hours: float | None = None, refresh: bool = False):
        """GET JSON, cached at cache_dir/<cache>. max_age_hours=None means the cache never expires."""
        path = self.cache_dir / cache
        if path.exists() and not refresh:
            stored = json.loads(path.read_text(encoding="utf-8"))
            # The download time is stored in the file: file times are unreliable (a CI cache restore
            # resets them). Older cache files without it fall back to the file time.
            if isinstance(stored, dict) and "_fetched_at" in stored:
                fetched, data = stored["_fetched_at"], stored["payload"]
            else:
                fetched, data = path.stat().st_mtime, stored
            age_h = (time.time() - fetched) / 3600
            if self.offline or max_age_hours is None or age_h < max_age_hours:
                return data
        if self.offline:
            raise ApiError(f"offline and no cached copy of {url} ({path})")
        data = self._download(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_fetched_at": time.time(), "payload": data}), encoding="utf-8")
        return data

    def _download(self, url: str, tries: int = 4):
        req = urllib.request.Request(url, headers={"User-Agent": self.settings.user_agent,
                                                   "Accept": "application/json"})
        for attempt in range(tries):
            wait = self.settings.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.requests += 1
            try:
                # urllib follows the 307 redirects the /now endpoints return.
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise ApiError(f"HTTP {e.code} for {url}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == tries - 1:
                    raise ApiError(f"{e} for {url}") from e
            time.sleep(2 ** attempt * 2)
        raise ApiError(f"giving up on {url}")

    # ---- endpoints

    def teams(self, refresh=False) -> list[dict]:
        return self.get(STATS + "team", "teams.json", max_age_hours=24 * 7, refresh=refresh)["data"]

    def franchises(self, refresh=False) -> list[dict]:
        return self.get(STATS + "franchise?include=teams", "franchises.json", max_age_hours=24 * 7,
                        refresh=refresh)["data"]

    def standings_seasons(self, refresh=False) -> list[dict]:
        return self.get(WEB + "standings-season", "standings-season.json", max_age_hours=24,
                        refresh=refresh)["seasons"]

    def season_games(self, season: int, current: bool, refresh=False) -> list[dict]:
        url = STATS + f"game?limit=-1&cayenneExp=season={season}"
        ttl = self.settings.current_ttl_hours if current else None
        return self.get(url, f"games/{season}.json", max_age_hours=ttl, refresh=refresh)["data"]

    def standings(self, date: str, season: int, current: bool, refresh=False) -> list[dict]:
        """Standings on `date` (YYYY-MM-DD); for the current season use 'now'."""
        url = WEB + ("standings/now" if current else f"standings/{date}")
        ttl = 6 if current else None
        return self.get(url, f"standings/{season}.json", max_age_hours=ttl, refresh=refresh)["standings"]
