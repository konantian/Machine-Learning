import xlsxwriter

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('test.xlsx')
worksheet = workbook.add_worksheet()

# Some data we want to write to the worksheet.
with open("Daily_Demand_Forecasting_Orders.csv","r") as file:
    file_data=file.read().splitlines()

temp=[]
for i in file_data:
	temp.append(i.split(";"))

# Start from the first cell. Rows and columns are zero indexed.
row = 0
# Iterate over the data and write it out row by row.
for data in temp:
	col=0
	for i in data:
		worksheet.write(row,col,i)
		col+=1
	row+=1
	#print(data)
	#worksheet.write(row, col,	 item)
	#worksheet.write(row, col + 1, cost)
	#row += 1

workbook.close()