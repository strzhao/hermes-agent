"""execute_code sandbox TZ inheritance (#138741).

``_build_child_env`` used to inject ``TZ`` only from the Hermes config chain
(``HERMES_TIMEZONE`` env / config.yaml ``timezone``). ``TZ`` is not in
``_SAFE_ENV_PREFIXES``, so a container deployed the conventional way —
``TZ=Asia/Shanghai`` in the environment, no Hermes timezone config — got a
UTC sandbox: the scrub dropped the parent's ``TZ`` and nothing re-injected
it, and execute_code ran hours off terminal/cron inside the same container.

Contract (three states, plus the #137153 invalid-config semantics):
1. a valid Hermes timezone config → the config wins (unchanged behavior);
2. no valid Hermes config (unset — or invalid: a bad configured value is not
   a deployment opinion) → the PARENT's ``TZ`` is inherited, like PATH/HOME/
   LANG describe the deployment locale;
3. neither → no ``TZ`` key at all (child uses the OS local time, as before).
"""

import os
import shutil
import sys
import tempfile
import unittest


@unittest.skipIf(sys.platform == "win32",
                 "Windows CPython does not take IANA names in TZ (children keep the OS zone)")
class TestSandboxTZInheritance(unittest.TestCase):
    """The sandbox child env must describe the deployment's timezone."""

    def setUp(self):
        import hermes_time

        self._hermes_time = hermes_time
        # Empty isolated Hermes home: the config chain (config.yaml) must read
        # as absent unless a test writes one — never the operator's real home.
        self._home = tempfile.mkdtemp(prefix="tz-inherit-home-")
        self._env_backup = {
            "HERMES_HOME": os.environ.pop("HERMES_HOME", None),
            "TZ": os.environ.pop("TZ", None),
            "HERMES_TIMEZONE": os.environ.pop("HERMES_TIMEZONE", None),
        }
        os.environ["HERMES_HOME"] = self._home
        hermes_time.reset_cache()
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in self._env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self._home, ignore_errors=True)
        self._hermes_time.reset_cache()

    def _child_env(self):
        from tools.code_execution_env import _build_child_env

        return _build_child_env(rpc_endpoint="/tmp/tz-test.sock", rpc_token="t",
                                tmpdir="/tmp/tz-test", child_python=sys.executable)

    def test_inherits_parent_tz_without_hermes_config(self):
        # Container deployment: TZ in the environment, no Hermes timezone config.
        os.environ["TZ"] = "Asia/Shanghai"
        self.assertEqual(self._child_env().get("TZ"), "Asia/Shanghai")

    def test_hermes_config_wins_over_parent_tz(self):
        # Explicit Hermes config stays authoritative (behavior guard, unchanged).
        os.environ["TZ"] = "Asia/Shanghai"
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        child_env = self._child_env()
        self.assertEqual(child_env.get("TZ"), "America/New_York")
        self.assertNotIn("HERMES_TIMEZONE", child_env)

    def test_no_tz_key_when_neither_configured(self):
        # Neither side names a zone → no TZ key (child uses the OS local time).
        self.assertNotIn("TZ", self._child_env())

    def test_invalid_config_falls_back_to_parent_tz(self):
        # #137153 semantics: an invalid configured value resolves to no zone,
        # which must reach the inheritance branch — never leak the bad name
        # into the child as TZ.
        with open(os.path.join(self._home, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write("timezone: Not/AZone\n")
        self._hermes_time.reset_cache()
        os.environ["TZ"] = "Asia/Shanghai"
        self.assertEqual(self._child_env().get("TZ"), "Asia/Shanghai")


if __name__ == "__main__":
    unittest.main()
