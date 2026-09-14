"""Shared environment-proxy routing for outbound HTTP clients."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass


LOCAL_PROXY_PROBE_TIMEOUT_SECONDS = 0.2


@dataclass(frozen=True)
class ProxyRouteDecision:
    """Resolved environment-proxy route for one outbound request."""

    trust_env: bool = True
    proxy_url: str | None = field(default=None, repr=False)
    proxy_address: str | None = None
    bypassed_unavailable_proxy: bool = False

    @property
    def cache_key(self) -> tuple[bool, str | None]:
        """Key used to reuse a client only while its proxy route is unchanged."""
        return self.trust_env, self.proxy_url


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _proxy_address(proxy_url: str) -> tuple[str, int, str] | None:
    normalized = proxy_url if "://" in proxy_url else f"http://{proxy_url}"
    try:
        parsed = urlsplit(normalized)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not host:
        return None
    if port is None:
        port = {
            "http": 80,
            "https": 443,
            "socks": 1080,
            "socks5": 1080,
            "socks5h": 1080,
        }.get(parsed.scheme.lower())
    if port is None:
        return None
    display_host = f"[{host}]" if ":" in host else host
    return host, port, f"{display_host}:{port}"


def detect_proxy_route(target_url: str | None) -> ProxyRouteDecision:
    """Ignore a stale local environment proxy while retaining valid proxies.

    Only loopback proxies are actively probed. Remote proxies are left to the
    HTTP client so this check never creates an additional external request.
    """
    try:
        parsed_target = urlsplit(target_url or "")
        target_host = parsed_target.hostname
        target_scheme = parsed_target.scheme.lower()
    except ValueError:
        return ProxyRouteDecision()
    if not target_host or target_scheme not in {"http", "https"}:
        return ProxyRouteDecision()

    try:
        if proxy_bypass(target_host):
            return ProxyRouteDecision()
    except (OSError, ValueError):
        pass

    try:
        proxies = getproxies()
    except OSError:
        return ProxyRouteDecision()
    proxy_url = proxies.get(target_scheme) or proxies.get("all")
    if not isinstance(proxy_url, str) or not proxy_url.strip():
        return ProxyRouteDecision()
    proxy_url = proxy_url.strip()
    address = _proxy_address(proxy_url)
    if address is None:
        return ProxyRouteDecision(trust_env=True, proxy_url=proxy_url)

    host, port, display = address
    if not _is_loopback_host(host):
        return ProxyRouteDecision(
            trust_env=True,
            proxy_url=proxy_url,
            proxy_address=display,
        )

    try:
        connection = socket.create_connection(
            (host, port), timeout=LOCAL_PROXY_PROBE_TIMEOUT_SECONDS,
        )
    except OSError:
        return ProxyRouteDecision(
            trust_env=False,
            proxy_url=proxy_url,
            proxy_address=display,
            bypassed_unavailable_proxy=True,
        )
    else:
        connection.close()
        return ProxyRouteDecision(
            trust_env=True,
            proxy_url=proxy_url,
            proxy_address=display,
        )
