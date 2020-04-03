FROM yondon/lpms-base:latest as builder

FROM python:3.6.7
RUN apt-get update
RUN apt-get install libev-dev -y
COPY --from=builder /root/compiled /root/compiled/

ENV PATH "$HOME/root/compiled/bin:$PATH"
ENV PKG_CONFIG_PATH "$HOME/root/compiled/lib/pkgconfig"

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY /scripts/asset_processor /scripts
COPY /verifier /scripts