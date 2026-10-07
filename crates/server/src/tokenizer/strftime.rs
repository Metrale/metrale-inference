// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded local-time strftime helper matching Hugging Face templates.
use minijinja::{Error, ErrorKind};

pub(super) fn now(format: String) -> Result<String, Error> {
    if format.len() > 256 {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "strftime format too long",
        ));
    }
    let mut chars = format.chars();
    while let Some(ch) = chars.next() {
        if ch == '%' && !chars.next().is_some_and(|c| "YmdHMSzZ%".contains(c)) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "unsupported strftime directive",
            ));
        }
    }
    let format = std::ffi::CString::new(format)
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "NUL in strftime format"))?;
    local_format(&format)
}

#[cfg(unix)]
fn local_format(format: &std::ffi::CStr) -> Result<String, Error> {
    // 2026-10-07: HF calls datetime.now().strftime: use process-local timezone, not UTC.
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "clock before Unix epoch"))?
        .as_secs();
    let seconds: libc::time_t = seconds
        .try_into()
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "clock overflow"))?;
    let mut calendar = std::mem::MaybeUninit::<libc::tm>::uninit();
    let mut output = [0u8; 4096];
    // SAFETY: localtime_r writes the caller-owned tm; strftime receives initialized tm,
    // a NUL-terminated format, and the exact writable output capacity.
    let size = unsafe {
        if libc::localtime_r(&seconds, calendar.as_mut_ptr()).is_null() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "local time unavailable",
            ));
        }
        libc::strftime(
            output.as_mut_ptr().cast(),
            output.len(),
            format.as_ptr(),
            calendar.as_ptr(),
        )
    };
    if size == 0 && !format.to_bytes().is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "strftime output unavailable or exceeds bound",
        ));
    }
    String::from_utf8(output[..size].to_vec())
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "strftime output is not UTF8"))
}

#[cfg(not(unix))]
fn local_format(_: &std::ffi::CStr) -> Result<String, Error> {
    Err(Error::new(
        ErrorKind::InvalidOperation,
        "checkpoint local-time template helper requires Unix",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generic_template_environment_does_not_gain_harmony_clock_global() {
        let env =
            super::super::jinja_helpers::build_jinja_env("{{ strftime_now is defined }}").unwrap();
        assert_eq!(
            env.get_template("chat").unwrap().render(()).unwrap(),
            "false"
        );
    }
    #[test]
    fn rejects_unbounded_and_nul_formats() {
        assert!(now("x".repeat(257)).is_err());
        assert!(now("%Y\0%m".into()).is_err());
        assert!(now("%Q".into()).is_err());
        assert!(now("%".into()).is_err());
    }
    #[test]
    #[cfg(unix)]
    fn supports_checkpoint_iso_date_and_escaped_percent() {
        let date = now("%Y-%m-%d".into()).unwrap();
        assert_eq!(date.len(), 10);
        assert_eq!(&date[4..5], "-");
        assert_eq!(&date[7..8], "-");
        assert_eq!(now("literal %%".into()).unwrap(), "literal %");
        assert_eq!(now(String::new()).unwrap(), "");
    }
}
