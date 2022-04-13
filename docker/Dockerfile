FROM python:3.9


RUN apt-get update && \
    apt-get install -y \
        file \
        libcurl4-gnutls-dev \
        libfreetype6-dev \
        libgmp-dev \
        libicu-dev \
        libjpeg62-turbo-dev \
        libmcrypt-dev \
        libmhash-dev \
        libonig-dev \
        libpng-dev \
        libsqlite3-dev \
        libzip-dev \
        re2c \
        cron


ENTRYPOINT [ "/entrypoint.sh" ]
