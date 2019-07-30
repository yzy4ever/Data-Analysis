import pandas as pd

#导入CSV文件
i=pd.read_csv("文件地址")
i.sort_values(by="列名")   #按照某列进行排序

#导入EXCEL文件
j=pd.read_excel("文件地址")

#导入数据库文件
import pymysql
conn=pymysql.connect(host="127.0.0.1",user="root",password="root",db="数据库名称")
sql="select* from student"
k=pd.read_sql(sql,conn)

#导入HTML表格数据
l=pd.read_html("文件地址或URL")

#导入文本文件
m=pd.read_table("文件地址")

#导出到EXCEL
file="./文件名称.xls"   #导出到项目当前的文件夹
j.to_excel(file,index=False)
