"""
Package Confusion Audit - Detecting malicious packages using statistical analysis.
Milestone 5.2 - Rishi's statistical analysis component
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)


def analyze_download_velocity(
    download_history: List[Dict[str, Any]],
    window_hours: int = 1,
) -> float:
    """
    Analyze download velocity (downloads per hour).

    Args:
        download_history: List of download events with 'downloaded_at' timestamps
        window_hours: Time window for analysis

    Returns:
        Downloads per hour (float)
    """
    if not download_history:
        return 0.0

    try:
        # Parse timestamps and find recent downloads
        now = datetime.now()
        window_start = now - timedelta(hours=window_hours)

        recent_downloads = [
            d
            for d in download_history
            if isinstance(d.get("downloaded_at"), str)
            and datetime.fromisoformat(d["downloaded_at"]) >= window_start
        ]

        if len(recent_downloads) == 0:
            return 0.0

        # Calculate downloads per hour
        velocity = len(recent_downloads) / window_hours
        logger.debug(f"Download velocity: {velocity} downloads/hour")

        return velocity

    except Exception as e:
        logger.error(f"Error analyzing download velocity: {e}")
        return 0.0


def calculate_user_diversity(download_history: List[Dict[str, Any]]) -> float:
    """
    Calculate user diversity metric (0-1 scale).

    Args:
        download_history: List of download events with 'downloader_username'

    Returns:
        User diversity score: unique_users / total_downloads (0-1)
    """
    if not download_history:
        return 1.0  # Perfect diversity if no downloads

    try:
        total_downloads = len(download_history)
        unique_users = len(set(d.get("downloader_username", "unknown") for d in download_history))

        diversity = unique_users / total_downloads
        logger.debug(f"User diversity: {diversity:.2f} ({unique_users}/{total_downloads})")

        return min(diversity, 1.0)

    except Exception as e:
        logger.error(f"Error calculating user diversity: {e}")
        return 1.0


def detect_bot_farm(
    download_history: List[Dict[str, Any]],
    indicators_threshold: int = 3,
) -> bool:
    """
    Detect bot farm patterns in download history.

    Indicators checked:
    - Rapid succession downloads (>5 in 60s)
    - Repeated username patterns
    - Suspicious timing patterns

    Args:
        download_history: List of download events
        indicators_threshold: Number of indicators to trigger bot detection

    Returns:
        True if bot farm detected, False otherwise
    """
    if len(download_history) < 5:
        return False

    try:
        indicators_detected = 0

        # Indicator 1: Rapid succession downloads
        if len(download_history) >= 5:
            recent_downloads = download_history[-5:]  # Last 5 downloads
            timestamps = []

            for d in recent_downloads:
                ts = d.get("downloaded_at")
                if isinstance(ts, str):
                    try:
                        timestamps.append(datetime.fromisoformat(ts))
                    except ValueError:
                        pass

            if len(timestamps) >= 5:
                time_diffs = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]
                avg_time_between = statistics.mean(time_diffs) if time_diffs else float("inf")

                # Suspicious if average < 2 seconds between downloads
                if avg_time_between < 2.0:
                    indicators_detected += 1
                    logger.warning(
                        f"Bot indicator: Rapid downloads ({avg_time_between:.1f}s apart)"
                    )

        # Indicator 2: Repeated username pattern
        usernames = [d.get("downloader_username", "unknown") for d in download_history]
        username_counts: defaultdict[str, int] = defaultdict(int)
        for u in usernames:
            username_counts[u] += 1

        # If one user downloads >50% of the time, suspicious
        if usernames:
            max_count = max(username_counts.values())
            if max_count > len(usernames) * 0.5:
                indicators_detected += 1
                logger.warning(
                    f"Bot indicator: One user dominated {max_count}/{len(usernames)} downloads"
                )

        # Indicator 3: Too many downloads from same uploader
        # (This would require uploader info in history - skip for now)

        is_bot_farm = indicators_detected >= indicators_threshold
        if is_bot_farm:
            logger.warning(f"Bot farm detected: {indicators_detected} indicators triggered")

        return is_bot_farm

    except Exception as e:
        logger.error(f"Error detecting bot farm: {e}")
        return False


def analyze_search_presence(
    model_name: str,
    search_hit_count: int = 1,
    total_searches: int = 100,
) -> float:
    """
    Analyze how often a model appears in search results.

    Args:
        model_name: Name of the model
        search_hit_count: Number of times found in searches
        total_searches: Total search queries executed

    Returns:
        Search presence score (0-1): hit_count / total_searches
    """
    try:
        if total_searches == 0:
            return 0.0

        presence = search_hit_count / total_searches
        logger.debug(f"Search presence for '{model_name}': {presence:.2f}")

        return min(presence, 1.0)

    except Exception as e:
        logger.error(f"Error analyzing search presence: {e}")
        return 0.0


def calculate_package_confusion_score(
    download_history: List[Dict[str, Any]],
    search_presence: float = 0.0,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Calculate overall package confusion risk score.

    Returns:
        {
            "suspicious": bool,
            "score": float (0-1),
            "reason": str,
            "indicators": {
                "velocity": float,
                "user_diversity": float,
                "bot_farm_detected": bool,
                "search_presence": float,
            }
        }
    """
    try:
        velocity = analyze_download_velocity(download_history, window_hours=1)
        user_diversity = calculate_user_diversity(download_history)
        is_bot_farm = detect_bot_farm(download_history)

        # Calculate risk score
        # Bot farm is strong signal (0.5 weight)
        # Low user diversity is concerning (0.15 weight)
        # High velocity is suspicious (0.15 weight)
        # Low search presence is suspicious (0.2 weight) - confusion attacks are less popular

        bot_farm_score = 0.5 if is_bot_farm else 0.0
        diversity_score = (1.0 - user_diversity) * 0.15  # Inverted: low diversity = high risk
        velocity_score = min(velocity / 10.0, 1.0) * 0.15  # Normalized to 10+ downloads/hour
        # Low search presence = higher risk (confusion attacks are less popular)
        search_presence_score = (1.0 - search_presence) * 0.2

        overall_score = min(
            bot_farm_score + diversity_score + velocity_score + search_presence_score, 1.0
        )

        # Determine if suspicious based on score
        sensitivity_threshold = 0.7
        is_suspicious = overall_score >= sensitivity_threshold

        # Generate reason
        reasons = []
        if is_bot_farm:
            reasons.append("Bot farm patterns detected")
        if user_diversity < 0.3:
            reasons.append("Low user diversity")
        if velocity > 5.0:
            reasons.append("High download velocity")
        if search_presence < 0.01:  # Less than 1% search presence
            reasons.append("Low search presence")

        reason = " | ".join(reasons) if reasons else "Clean"

        return {
            "suspicious": is_suspicious,
            "score": overall_score,
            "reason": reason,
            "indicators": {
                "velocity": velocity,
                "user_diversity": user_diversity,
                "bot_farm_detected": is_bot_farm,
                "search_presence": search_presence,
            },
        }

    except Exception as e:
        logger.error(f"Error calculating package confusion score: {e}")
        return {
            "suspicious": False,
            "score": 0.0,
            "reason": "Error analyzing package",
            "indicators": {
                "velocity": 0.0,
                "user_diversity": 1.0,
                "bot_farm_detected": False,
                "search_presence": 0.0,
            },
        }
