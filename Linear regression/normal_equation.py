import numpy as np
from numpy.linalg import pinv
from scipy import stats

def main():
	filename=input("Please enter the file name: ")
	#filename="ex1data2.txt"
	with open(filename,"r") as file:
		file_data=file.read().splitlines()

	process_data(file_data)

def process_data(file_data):
	x=[];y=[];lable_x=[];lable_y=[];

	choice=1

	for i in file_data:
		data=i.split(",")
		size=len(data)
		choice = 1 if size == 2 else 0
		temp=[]
		temp.append(1)
		for m in range(len(data)-1):
			temp.append(float(data[m]))
		x.append(temp)
		y.append([float(data[-1])])

		lable_x.append(float(data[0]))
		lable_y.append(float(data[-1]))

	x=np.array(x)
	x_T=x.transpose()
	y=np.array(y)
	lable_x=np.array(lable_x)
	lable_y=np.array(lable_y)

	if choice == 0:
	 	result=matrix_multiplication(x,x_T,y)
	 	result=least_square(x,y)
	 	output(result,size)

	elif choice == 1:
	 	linear_regression(lable_x,lable_y)
	 	gradient_descent(lable_x,lable_y)
	
#first way to do this for multiple variables
def matrix_multiplication(x,x_T,y):

	mid=np.matmul(pinv(np.matmul(x_T,x)),x_T) #(x^T*X)^-1*X^T
	print(np.shape(mid),np.shape(y))
	result=np.matmul(mid,y) #(x^T*X)^-1*X^T * y

	return result

#second way to do this for multiple variables
def least_square(x,y):

	result = np.linalg.lstsq(x, y,rcond=None)[0]

	return result

#third way to do this for one variables,linear regression
def linear_regression(x,y):

	slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y) 

	print('intercept = %s slope = %s' % (round(intercept,2), round(slope,2)))
	plot_graph(x,y,slope,intercept)
	return slope,intercept

def gradient_descent(x,target_data):
	m=len(x)
	print(m)
	x0=np.full(m,1.0)
	input_data=np.vstack([x0,x]).T


	# 两种终止条件  
	loop_max = 100000  # 最大迭代次数(防止死循环)  
	epsilon = 1e-6  
	  
	# 初始化权值  
	np.random.seed(0)  
	theta = np.random.randn(2)  
	  
	alpha = 0.0001  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)  
	diff = 0.  
	error = np.zeros(2)  
	count = 0  # 循环次数  
	finish = 0  # 终止标志  
	  
	while count < loop_max:  
		count += 1  
	  
		# 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的  
		# 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算  
		sum_m = np.zeros(2)  
		for i in range(m):  
			dif = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]  
			sum_m = sum_m + dif  # 当alpha取值过大时,sum_m会在迭代过程中会溢出  
	  
		theta = theta - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡  
		# theta = theta - 0.005 * sum_m	  # alpha取0.005时产生振荡,需要将alpha调小  
	  
		# 判断是否已收敛  
		if np.linalg.norm(theta - error) < epsilon:  
			finish = 1  
			break  
		else:  
			error = theta  
		#print('loop count = %d' % count, '\tw:',theta)  
	print('loop count = %d' % count, '\tw:',theta) 
	#plot_graph(x,target_data,theta[1],theta[0])

def plot_graph(x,y,slope,intercept):
	import matplotlib.pyplot as plt  
	plt.plot(x, y,"g*") 
	plt.plot(x, slope * x + intercept, 'r') 
	plt.legend()
	plt.show()

def output(result,size):

	print("Constant is: "+str(round(result[0][0])))
	print("x1 to x"+str(size-1)+" is: ")
	for i in range(1,size):
		print(round(result[i][0],2))

main()



