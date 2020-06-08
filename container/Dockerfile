FROM python:3.6

# set the working directory
# RUN ["mkdir", "app"]
WORKDIR /opt/program

# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

ADD input /opt/program/input
ADD model /opt/program/model
ADD output /opt/program/output

# install environment dependencies
COPY "app.py" .
COPY "run.sh" .
# COPY "train" .
COPY "train.py" .
# COPY "health-check-data.csv" .

# provision environment
ENV FLASK_APP app.py
RUN ["chmod", "+x", "./run.sh"]
EXPOSE 8080
ENTRYPOINT ["./run.sh"]
CMD ["train"]

ENV PATH="/opt/program:${PATH}"


