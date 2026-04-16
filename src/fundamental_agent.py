from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
from typing import Any

import requests


@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    url: str
    published_utc: datetime | None


@dataclass
class FundamentalDecision:
    action: str
    confidence: float
    rationale: str
    headlines_used: int
    analysis_source: str = "unknown"
    news_signature: str = ""
    news_changed: bool = False


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    # Fast path for strict JSON-only responses.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Common model behavior: wrap JSON in markdown code fences.
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Last resort: find a JSON object span in free-form text.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _parse_published_utc(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class RssNewsProvider:
    def __init__(self, timeout_seconds: int = 8, user_agent: str = "EconomicAE/1.0 (+research)"):
        self.timeout_seconds = max(2, int(timeout_seconds))
        self.user_agent = str(user_agent)

    def fetch(self, url: str, max_items: int = 8) -> list[NewsItem]:
        if not url:
            return []
        headers = {"User-Agent": self.user_agent, "Accept": "application/rss+xml, application/xml, text/xml, */*"}
        r = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        items: list[NewsItem] = []
        source = _extract_hostname(url)

        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            summary = (node.findtext("description") or "").strip()
            pub_raw = (node.findtext("pubDate") or node.findtext("published") or "").strip()
            pub_utc = _parse_published_utc(pub_raw)
            if not title:
                continue
            items.append(
                NewsItem(
                    source=source,
                    title=_clean_html(title),
                    summary=_clean_html(summary),
                    url=link,
                    published_utc=pub_utc,
                )
            )
            if len(items) >= max_items:
                break

        return items


class OpenAICompatibleFundamentalLLM:
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        temperature: float,
        max_tokens: int,
    ):
        self.api_base_url = str(api_base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "gpt-4o-mini")
        self.timeout_seconds = max(4, int(timeout_seconds))
        self.temperature = float(max(0.0, min(1.0, temperature)))
        self.max_tokens = max(120, int(max_tokens))

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def analyze(
        self,
        symbol: str,
        asset_class: str,
        headlines: list[NewsItem],
        event_context: dict[str, Any] | None = None,
    ) -> FundamentalDecision | None:
        if not self.available() or not headlines:
            return None

        bullets = []
        for h in headlines:
            ts = h.published_utc.isoformat() if h.published_utc else "n/a"
            bullets.append(f"- [{h.source}] {h.title} (published={ts})")

        system_prompt = (
            "You are a macro/fundamental trading analyst. "
            "Return ONLY valid JSON with keys: action, confidence, rationale. "
            "action must be BUY, SELL or HOLD. confidence in [0,1]. "
            "Be conservative when news is mixed or weak."
        )
        event_name = str((event_context or {}).get("name", "")).strip()
        event_currency = str((event_context or {}).get("currency", "")).strip().upper()
        event_importance = str((event_context or {}).get("importance", "")).strip()
        event_block = ""
        if event_name or event_currency or event_importance:
            event_block = (
                "Primary scheduled event context:\n"
                f"- name: {event_name or 'n/a'}\n"
                f"- currency: {event_currency or 'n/a'}\n"
                f"- importance: {event_importance or 'n/a'}\n"
            )

        user_prompt = (
            f"Asset symbol: {symbol}\n"
            f"Asset class: {asset_class}\n"
            + event_block
            +
            "Recent macro/news headlines:\n"
            + "\n".join(bullets)
            + "\nDecide directional action for the next short horizon."
        )

        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            obj = _extract_json_object(text) if isinstance(text, str) else None
            if obj is None:
                return None
            action = str(obj.get("action", "HOLD")).strip().upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                action = "HOLD"
            conf = max(0.0, min(1.0, _safe_float(obj.get("confidence", 0.5), 0.5)))
            rationale = str(obj.get("rationale", ""))[:500]
            return FundamentalDecision(action=action, confidence=conf, rationale=rationale, headlines_used=len(headlines))
        except Exception:
            return None


class FundamentalNewsLLMEngine:
    def __init__(self, settings):
        self.lookback_minutes = max(30, int(getattr(settings, "fundamental_news_lookback_minutes", 240)))
        self.news_poll_seconds = max(0, int(getattr(settings, "fundamental_news_poll_seconds", 20)))
        self.max_headlines = max(5, int(getattr(settings, "fundamental_max_headlines", 30)))
        self.max_headlines_per_source = max(2, int(getattr(settings, "fundamental_max_headlines_per_source", 8)))
        self.use_heuristic_fallback = bool(getattr(settings, "fundamental_use_heuristic_fallback", True))
        self.reanalyze_seconds = max(0, int(getattr(settings, "fundamental_reanalyze_seconds", 15)))

        raw_sources = str(getattr(settings, "fundamental_news_sources", "")).strip()
        self.news_sources = [x.strip() for x in raw_sources.split(",") if x.strip()]

        self.provider = RssNewsProvider(
            timeout_seconds=int(getattr(settings, "fundamental_news_timeout_seconds", 8)),
            user_agent=str(getattr(settings, "fundamental_user_agent", "EconomicAE/1.0 (+research)")),
        )
        fundamental_api_key = str(getattr(settings, "fundamental_llm_api_key", "")).strip()
        if fundamental_api_key:
            api_base_url = str(getattr(settings, "fundamental_llm_api_base_url", "https://api.openai.com/v1")).strip()
            api_key = fundamental_api_key
            model = str(getattr(settings, "fundamental_llm_model", "gpt-4o-mini")).strip()
        else:
            # Gemini fallback path (OpenAI-compatible endpoint)
            api_base_url = str(getattr(settings, "gemini_openai_base_url", "https://generativelanguage.googleapis.com/v1beta/openai")).strip()
            api_key = str(getattr(settings, "gemini_api_key", "")).strip()
            model = str(getattr(settings, "gemini_model", "gemini-3.1-pro-preview")).strip()

        self.llm = OpenAICompatibleFundamentalLLM(
            api_base_url=api_base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=int(getattr(settings, "fundamental_llm_timeout_seconds", 12)),
            temperature=float(getattr(settings, "fundamental_llm_temperature", 0.1)),
            max_tokens=int(getattr(settings, "fundamental_llm_max_tokens", 250)),
        )
        self._last_news_signature: str = ""
        self._last_decision: FundamentalDecision | None = None
        self._last_analysis_utc: datetime | None = None
        self._last_headlines: list[NewsItem] = []
        self._last_headlines_fetch_utc: datetime | None = None
        self._stale_decision_max_age_seconds: int = 180

    def analyze(self, symbol: str, event_context: dict[str, Any] | None = None) -> FundamentalDecision:
        asset_class = _infer_asset_class(symbol)
        headlines = self._get_recent_headlines_cached()
        signature = self._headlines_signature(headlines)
        news_changed = bool(signature and signature != self._last_news_signature)
        now_utc = datetime.now(timezone.utc)

        if (
            not news_changed
            and self._last_decision is not None
            and self._last_analysis_utc is not None
            and (now_utc - self._last_analysis_utc).total_seconds() < float(self.reanalyze_seconds)
        ):
            d = self._last_decision
            return FundamentalDecision(
                action=d.action,
                confidence=d.confidence,
                rationale=d.rationale,
                headlines_used=len(headlines),
                analysis_source="cache",
                news_signature=signature,
                news_changed=False,
            )

        llm_out = self.llm.analyze(
            symbol=symbol,
            asset_class=asset_class,
            headlines=headlines,
            event_context=event_context,
        )
        if llm_out is not None:
            decision = FundamentalDecision(
                action=llm_out.action,
                confidence=llm_out.confidence,
                rationale=llm_out.rationale,
                headlines_used=len(headlines),
                analysis_source="llm",
                news_signature=signature,
                news_changed=news_changed,
            )
            self._last_news_signature = signature
            self._last_decision = decision
            self._last_analysis_utc = now_utc
            return decision

        if self.use_heuristic_fallback:
            if not headlines and self._last_decision is not None and self._last_analysis_utc is not None:
                age = float((now_utc - self._last_analysis_utc).total_seconds())
                if age <= float(self._stale_decision_max_age_seconds):
                    d = self._last_decision
                    return FundamentalDecision(
                        action=d.action,
                        confidence=float(max(0.0, min(1.0, d.confidence * 0.9))),
                        rationale=(d.rationale or "")[:450] + " | stale_decision_fallback",
                        headlines_used=len(headlines),
                        analysis_source="stale_cache",
                        news_signature=signature,
                        news_changed=False,
                    )

            heuristic = _heuristic_fundamental_decision(headlines, symbol=symbol, event_context=event_context)
            decision = FundamentalDecision(
                action=heuristic.action,
                confidence=heuristic.confidence,
                rationale=heuristic.rationale,
                headlines_used=len(headlines),
                analysis_source="heuristic",
                news_signature=signature,
                news_changed=news_changed,
            )
            self._last_news_signature = signature
            self._last_decision = decision
            self._last_analysis_utc = now_utc
            return decision

        decision = FundamentalDecision(
            action="HOLD",
            confidence=0.0,
            rationale="No LLM response and fallback disabled.",
            headlines_used=len(headlines),
            analysis_source="none",
            news_signature=signature,
            news_changed=news_changed,
        )
        self._last_news_signature = signature
        self._last_decision = decision
        self._last_analysis_utc = now_utc
        return decision

    def _get_recent_headlines_cached(self) -> list[NewsItem]:
        now_utc = datetime.now(timezone.utc)
        if self.news_poll_seconds > 0 and self._last_headlines_fetch_utc is not None:
            elapsed = float((now_utc - self._last_headlines_fetch_utc).total_seconds())
            if elapsed < float(self.news_poll_seconds):
                return list(self._last_headlines)

        headlines = self._fetch_recent_headlines()
        if headlines:
            self._last_headlines = list(headlines)
        elif self._last_headlines:
            # Keep the most recent successful snapshot when feeds fail transiently.
            headlines = list(self._last_headlines)
        self._last_headlines_fetch_utc = now_utc
        return headlines

    def _fetch_recent_headlines(self) -> list[NewsItem]:
        if not self.news_sources:
            return []
        now_utc = datetime.now(timezone.utc)
        min_ts = now_utc - timedelta(minutes=self.lookback_minutes)
        items: list[NewsItem] = []

        for src in self.news_sources:
            try:
                rows = self.provider.fetch(src, max_items=self.max_headlines_per_source)
            except Exception:
                continue
            for row in rows:
                if row.published_utc is not None and row.published_utc < min_ts:
                    continue
                items.append(row)

        items.sort(key=lambda x: x.published_utc or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
        return items[: self.max_headlines]

    @staticmethod
    def _headlines_signature(headlines: list[NewsItem]) -> str:
        if not headlines:
            return ""
        normalized = []
        for h in headlines:
            ts = h.published_utc.isoformat() if h.published_utc else ""
            normalized.append(f"{h.source}|{h.title}|{ts}")
        joined = "\n".join(normalized)
        return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()


def _extract_hostname(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", str(url).strip().lower())
    return m.group(1) if m else "unknown"


def _clean_html(text: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(text or ""))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _infer_asset_class(symbol: str) -> str:
    s = str(symbol or "").upper()
    if len(s) == 6 and s.isalpha():
        return "forex"
    if any(x in s for x in ["XAU", "XAG", "WTI", "BRENT", "GOLD", "SILVER", "OIL"]):
        return "commodity"
    if any(x in s for x in ["US500", "NAS", "DJ", "DAX", "NIK", "SPX", "NDX", "ES", "NQ", "YM"]):
        return "index_or_future"
    if "." in s or "#" in s:
        return "equity"
    if len(s) <= 5 and s.isalpha():
        return "equity"
    return "multi_asset"


def _event_currency_prior(symbol: str, event_context: dict[str, Any] | None) -> str | None:
    s = str(symbol or "").upper().strip()
    ccy = str((event_context or {}).get("currency", "")).upper().strip()
    if len(s) == 6 and s.isalpha() and len(ccy) == 3:
        base = s[:3]
        quote = s[3:]
        if ccy == base:
            return "BUY"
        if ccy == quote:
            return "SELL"
    return None


def _heuristic_fundamental_decision(
    headlines: list[NewsItem],
    symbol: str = "",
    event_context: dict[str, Any] | None = None,
) -> FundamentalDecision:
    prior = _event_currency_prior(symbol=symbol, event_context=event_context)

    if not headlines:
        if prior in {"BUY", "SELL"}:
            return FundamentalDecision(
                action=prior,
                confidence=0.62,
                rationale=f"No headlines; event-currency prior={prior}.",
                headlines_used=0,
            )
        return FundamentalDecision(action="HOLD", confidence=0.0, rationale="No recent headlines found.", headlines_used=0)

    text = " ".join((h.title + " " + h.summary) for h in headlines).lower()
    positive = ["beat", "growth", "surge", "bullish", "rate cut", "strong jobs", "de-escalation", "upgrade"]
    negative = ["miss", "recession", "selloff", "bearish", "rate hike", "inflation spike", "war", "downgrade"]

    pos = sum(text.count(k) for k in positive)
    neg = sum(text.count(k) for k in negative)

    total = max(1, pos + neg)
    edge = abs(pos - neg) / total
    confidence = max(0.45, min(0.80, 0.45 + edge * 0.35))

    if pos == neg:
        if prior in {"BUY", "SELL"}:
            return FundamentalDecision(
                action=prior,
                confidence=0.60,
                rationale=f"Mixed sentiment; event-currency prior={prior}.",
                headlines_used=len(headlines),
            )
        return FundamentalDecision(action="HOLD", confidence=0.50, rationale="Mixed macro sentiment.", headlines_used=len(headlines))
    action = "BUY" if pos > neg else "SELL"
    return FundamentalDecision(action=action, confidence=confidence, rationale=f"Heuristic sentiment pos={pos} neg={neg}.", headlines_used=len(headlines))
