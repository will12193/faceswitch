FROM python:3.9 as production

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        vim

COPY requirements.txt requirements.txt 
COPY scripts/entrypoint.sh /entrypoint.sh

RUN pip install -r ./requirements.txt 



ENTRYPOINT [ "/entrypoint.sh" ]
