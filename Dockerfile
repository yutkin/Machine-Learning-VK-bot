FROM python:3.7

ENV TZ Europe/Moscow

ADD requirements.txt /app/
RUN pip install -U pip -r /app/requirements.txt

RUN pip install -U git+https://github.com/Supervisor/supervisor \
    git+https://github.com/MagicStack/uvloop

ADD setup.py /app/
ADD etc /app/config/
ADD vkbot /app/vkbot/
ADD tests /app/tests/

RUN pip install -e /app

EXPOSE 8080

WORKDIR /app
