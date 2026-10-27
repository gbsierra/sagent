## 2026-10-27 - Safe File Opening and Symlink Attacks
**Learning:** `os.open` without `O_EXCL` in a shared location is vulnerable to symlink attacks if `O_NOFOLLOW` is not specified. CodeQL and other security linters flag this as an arbitrary execute result (or file overwrite) vulnerability.
**Action:** Always append `| getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)` to `os.open` flags when `O_CREAT | O_APPEND` is used without `O_EXCL` in a potentially user-accessible directory to ensure explicit tight controls.
