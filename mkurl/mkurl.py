#encoding:utf-8  

import urllib2  

if __name__ == '__main__':  
    url = 'http://www.baidu.com'  
	try:
		res = urllib2.urlopen(url)  
		res_data = res.read()  
		print res_data 
    except urllib2.HTTPError, e
        print e.code
    except urllib2.URLError, e
        print str(e)