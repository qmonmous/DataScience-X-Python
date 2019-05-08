# Start with your own Virtual Machine

We are going to build a Big Data environment using HDFS with Hadoop, Hive and Spark.

### Simulate an Hadoop environment in a VM

Download VirtualBox.  
Download the VM and add it.  
Launch the VM and complete both login and password with `hadoop`.  
Now run the following commands in its terminal:  
- `start-dfs.sh` to launch the data file system.  
- `start-yarn.sh` to launch the yarn ressources manager.  
- `jps` to handle Java processes.  

Now you’ll need VM’s _ip_, run `ip addr show` or `ifconfig`.  
Copy it so you can access the web interface at this url: _ip:50070_ (HDFS) or _ip:8088_ (Yarn Ressource Manager).  

> Tips: You can interact with the VM through your own terminal by running `ssh hadoop@ip` in the terminal. 

#### Close the VM

If you want to close your VM, run:  
- `stop-yarn.sh`  
- `stop-dfs.sh`  
- `sudo poweroff`  


## II. Build a Hive database


Apache Hive can be considered as a neo SQL, designed for Big Data. /// ... blablabla ... ///

- come back to the root: `hdfs dfs -ls /`, then run: `cd public`
- create a repository for your data: `hdfs dfs -mkdir /user/hadoop/data`
- move a file from the current repository to another: `hdfs dfs -put dat_svi_data.csv /user/hadoop/data/`
- launch Hive: `hive`
- create a database: `CREATE DATABASES;`

```sql
Create data structure scheme:
CREATE TABLE svi_data (
    calldate string,
    calltime string,
    callid bigint,
    calltype string,
    calltype2 string,
    phonenumber string,
    userid bigint)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\;';

LOAD DATA INPATH '/user/hadoop/data/dat_svi_data.csv' INTO TABLE svi_data;
```

Now you can try different SQL requests:
```sql
SELECT * FROM svi_data LIMIT 10;
```
```sql
SELECT COUNT(*) FROM svi_data;
```


## III. Try pySpark

Let's move on Spark. We'll try this tool by practicing in a jupyter notebook. Run `cd notebooks` to move to the notebooks directory, then `jupyter notebook` to build the environment.  
Now open a browser and go to the following url: *ip:9090*. Again, the password is `hadoop`. This will give you a web interface for Spark at ip:4042.  

```python
from pyspark import SparkConf, SparkContext, HiveContext
```

```python
conf = SparkConf().setMaster('local[*]')
conf = conf.setAppName('APPTEST')
conf = conf.set('spark.ui.port', '4042')
sc = SparkContext(conf=conf)
hctx = HiveContext(sc)
hctx.sql('SHOW DATABASES').show()
```