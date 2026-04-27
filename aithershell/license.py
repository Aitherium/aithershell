"""
AitherShell License Key Validation
===================================

HMAC-SHA256 based license key validation for AitherShell.

License key format:
  {tier}:{user_id}:{expiry}:{signature}
  
  Example:
    free:user123:2026-12-31:abc123def456...
    pro:org456:unlimited:ghi789jkl012...

Tiers:
  - free: 5 queries/day, limited features
  - pro: Unlimited queries, team features, $9/month
  - enterprise: Custom, self-hosted, SLAs

Environment variable: AITHERIUM_LICENSE_KEY
"""

import hashlib
import hmac
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Signing secret (should be in environment or config)
LICENSE_SECRET = os.getenv("AITHERIUM_LICENSE_SECRET", "aitherium-default-secret-key-v1")


class LicenseError(Exception):
    """License validation error."""
    pass


class License:
    """AitherShell license key."""

    def __init__(self, tier: str, user_id: str, expiry: str, signature: str):
        """Initialize license.
        
        Args:
            tier: License tier (free, pro, enterprise)
            user_id: User or organization ID
            expiry: Expiration date (YYYY-MM-DD or 'unlimited')
            signature: HMAC-SHA256 signature
        """
        self.tier = tier
        self.user_id = user_id
        self.expiry = expiry
        self.signature = signature

    def is_valid(self) -> bool:
        """Check if license is valid."""
        # Check signature
        expected_sig = self._compute_signature()
        if not hmac.compare_digest(self.signature, expected_sig):
            return False

        # Check expiry
        if self.expiry != "unlimited":
            try:
                expiry_date = datetime.strptime(self.expiry, "%Y-%m-%d")
                if datetime.now() > expiry_date:
                    return False
            except ValueError:
                return False

        return True

    def is_expired(self) -> bool:
        """Check if license is expired."""
        if self.expiry == "unlimited":
            return False
        try:
            expiry_date = datetime.strptime(self.expiry, "%Y-%m-%d")
            return datetime.now() > expiry_date
        except ValueError:
            return False

    def days_until_expiry(self) -> Optional[int]:
        """Days until license expires (None if unlimited)."""
        if self.expiry == "unlimited":
            return None
        try:
            expiry_date = datetime.strptime(self.expiry, "%Y-%m-%d")
            delta = expiry_date - datetime.now()
            return max(0, delta.days)
        except ValueError:
            return None

    def _compute_signature(self) -> str:
        """Compute HMAC-SHA256 signature."""
        message = f"{self.tier}:{self.user_id}:{self.expiry}"
        signature = hmac.new(
            LICENSE_SECRET.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    @staticmethod
    def parse(key_string: str) -> "License":
        """Parse license key string.
        
        Args:
            key_string: License key (format: tier:user_id:expiry:signature)
            
        Returns:
            License instance
            
        Raises:
            LicenseError: If key format is invalid
        """
        parts = key_string.strip().split(":")
        if len(parts) != 4:
            raise LicenseError("Invalid license key format (expected 4 parts)")

        tier, user_id, expiry, signature = parts

        # Validate tier
        if tier not in ("free", "pro", "enterprise"):
            raise LicenseError(f"Invalid tier: {tier}")

        return License(tier, user_id, expiry, signature)

    def __str__(self) -> str:
        """Return license key string."""
        return f"{self.tier}:{self.user_id}:{self.expiry}:{self.signature}"

    def __repr__(self) -> str:
        """Return license info (redacted)."""
        return f"License(tier={self.tier}, user={self.user_id[:4]}..., expiry={self.expiry})"


def load_license() -> Optional[License]:
    """Load license from environment or config file.
    
    Returns:
        License instance or None if not found
        
    Raises:
        LicenseError: If license is invalid
    """
    # Try environment variable first
    key_string = os.getenv("AITHERIUM_LICENSE_KEY")

    if not key_string:
        # Try config file
        config_path = Path.home() / ".aither" / "license.key"
        if config_path.exists():
            key_string = config_path.read_text().strip()

    if not key_string:
        return None

    return License.parse(key_string)


def validate_license() -> Tuple[bool, str]:
    """Validate current license.
    
    Returns:
        (is_valid, message) tuple
    """
    try:
        license_obj = load_license()

        if not license_obj:
            return (
                False,
                "No license key found. Get one at https://aitherium.com/free",
            )

        if not license_obj.is_valid():
            return (False, f"Invalid license key: {license_obj}")

        if license_obj.is_expired():
            return (False, f"License expired on {license_obj.expiry}")

        # Check if expiring soon (pro/enterprise only)
        days_left = license_obj.days_until_expiry()
        if days_left is not None and days_left < 7 and license_obj.tier in ("pro", "enterprise"):
            return (
                True,
                f"⚠️  License expires in {days_left} days ({license_obj.expiry})",
            )

        return (True, f"License valid ({license_obj.tier} tier)")

    except LicenseError as e:
        return (False, f"License error: {e}")


def enforce_license(required_tier: str = "free") -> None:
    """Enforce license requirement.
    
    Args:
        required_tier: Minimum required tier (free, pro, enterprise)
        
    Raises:
        LicenseError: If license is invalid or insufficient
    """
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}

    try:
        license_obj = load_license()

        if not license_obj:
            raise LicenseError(
                "License required. Get one at https://aitherium.com/free"
            )

        if not license_obj.is_valid():
            raise LicenseError(f"Invalid license key")

        if license_obj.is_expired():
            raise LicenseError(f"License expired on {license_obj.expiry}")

        # Check tier level
        if tier_levels.get(license_obj.tier, -1) < tier_levels.get(required_tier, 0):
            raise LicenseError(
                f"License tier '{license_obj.tier}' insufficient. "
                f"Requires '{required_tier}' tier."
            )

    except LicenseError:
        raise


def get_tier() -> str:
    """Get current license tier.
    
    Returns:
        Tier name or 'unlicensed'
    """
    try:
        license_obj = load_license()
        if license_obj and license_obj.is_valid():
            return license_obj.tier
    except LicenseError:
        pass

    return "unlicensed"


# Example usage
if __name__ == "__main__":
    # Check license
    is_valid, message = validate_license()
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")

    if not is_valid:
        sys.exit(1)
