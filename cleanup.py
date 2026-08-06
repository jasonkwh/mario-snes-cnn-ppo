def close_env(env):
    if env is not None:
        try:
            env.close()
        except Exception:
            pass

