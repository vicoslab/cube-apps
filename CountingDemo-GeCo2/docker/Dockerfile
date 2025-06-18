FROM vicoslab/echolib:22.04 as echo
FROM counting-geco2:test

WORKDIR /app

COPY --from=echo /usr/local/lib /opt/venv/lib
COPY --chown=user echolib_wrapper.py ./
COPY --chown=user run_main.py ./

ENTRYPOINT ["/opt/venv/bin/python3", "run_main.py"]
