FROM python:3.6

RUN mkdir -p /home/app/
RUN rm -r /home/app
COPY ./static /home/app/static
COPY ./templates /home/app/templates
COPY ./docker_requirements.txt /home/app/
COPY ./app_utils.py /home/app/
COPY ./app.py /home/app/
WORKDIR /home/app
RUN pip3 install --upgrade pip
RUN pip3 install -r docker_requirements.txt

CMD python3 app.py