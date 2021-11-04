FROM tensorflow/tensorflow:latest-gpu

# Spark dependencies
ENV APACHE_SPARK_VERSION 3.1.2
ENV HADOOP_VERSION 3.2

# Update os and install java
RUN apt-get update && apt-get install -y \
    locales \
    ca-certificates \
    software-properties-common \
    gpg \
    htop tmux wget git

# Get jdk11
RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && \
    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb && \
    apt-get update && apt-get install -y adoptopenjdk-11-hotspot

WORKDIR /docker_setting

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    wget https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz.asc && \
    wget https://archive.apache.org/dist/spark/KEYS && \
    gpg --import KEYS && gpg --verify spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz.asc

RUN mkdir /temp && mv spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz /temp && cd /temp && \
    tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local --owner work --group root --no-same-owner && \
    rm spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

# Install AWS S3 jars
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.0/hadoop-aws-3.2.0.jar && \
    mv hadoop-aws-3.2.0.jar /usr/local/spark/jars/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.25/aws-java-sdk-bundle-1.12.25.jar && \
    mv aws-java-sdk-bundle-1.12.25.jar /usr/local/spark/jars/ && \
    wget https://repository.cloudera.com/artifactory/cloudera-repos/org/apache/spark/spark-hadoop-cloud_2.12/3.1.1.3.1.7270.0-253/spark-hadoop-cloud_2.12-3.1.1.3.1.7270.0-253.jar && \
    mv spark-hadoop-cloud_2.12-3.1.1.3.1.7270.0-253.jar /usr/local/spark/jars/

# Install other dependencies
RUN wget https://repos.spark-packages.org/graphframes/graphframes/0.8.1-spark3.0-s_2.12/graphframes-0.8.1-spark3.0-s_2.12.jar && \
    mv graphframes-0.8.1-spark3.0-s_2.12.jar /usr/local/spark/jars/ && \
    wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.16/slf4j-api-1.7.16.jar && \
    mv slf4j-api-1.7.16.jar /usr/local/spark/jars/ && \
    wget https://repo1.maven.org/maven2/com/linkedin/sparktfrecord/spark-tfrecord_2.12/0.3.3/spark-tfrecord_2.12-0.3.3.jar && \
    mv spark-tfrecord_2.12-0.3.3.jar /usr/local/spark/jars/

# Spark config
COPY ./spark-defaults.conf /usr/local/spark/conf/
ENV SPARK_HOME /usr/local/spark
ENV PYTHONPATH $SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip

COPY . /app
WORKDIR /app

ENTRYPOINT bash
