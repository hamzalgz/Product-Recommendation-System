import os
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup

width = os.get_terminal_size().columns
print("\n\n\n" +"Amazon WebScraping".center(width))


print("Enter Amazon site url without page number\n\n")

inp = input("Example: https://www.amazon.in/s/rh=n%3A976419031%2Cn%3A%21976420031%2Cn%3A1389401031&page=\n\n\n")
print(inp)
pg = input("Enter the number of pages to scrap\n\n")

for a in range(1,int(pg)+1):
	print("Acessing page " + str(a))

	temp_url = pg

	my_url = temp_url + str(a)
	req = Request(my_url, headers={'User-Agent': 'Mozilla/5.0'})
	page_html = urlopen(req).read()    

	page_soup = soup(page_html, "html.parser")   

	containers = page_soup.findAll("div",{"class":"s-item-container"})

	filename = "Amazon products.csv"
	f = open(filename,"a" , encoding='utf-8')   

	headers ="brand , proudct_name , retail_price , price , rating,  offer\n"
	f.write(headers)

	for container in containers:
		product_name = container.h2['data-attribute'].strip()
    	
		price_container = container.findAll("span",{"class":"a-size-base a-color-price s-price a-text-bold"})
		try:
			price = price_container[0].text.strip()
		except IndexError:
			price = 'null'

		Rating_container = container.findAll("span",{"class":"a-icon-alt"})
		try:
			rating = Rating_container[1].text.strip()
		except IndexError:
			rating = 'null'
		
		brand_container = container.findAll("span",{"class":"a-size-small a-color-secondary"})
		try:
			brand = brand_container[1].text.strip()
		except IndexError:
			brand = "null"
		
		ret_container = container.findAll("span",{"class":"a-size-small a-color-secondary a-text-strike"})
		try:
			retail_price = ret_container[0].text.strip()
		except IndexError:
			retail_price = "null"

		offer_container = container.findAll("span",{"class":"a-list-item"})
		offer = ""
		for i in range(10):
			try:
				offer = str(offer) + " " +offer_container[i].text.strip()
			except IndexError:
				break

		print("Brand         :" + brand)
		print("Product       :" + product_name + "\n")
		print("Orginal price :" + retail_price)
		print("Current price :" + price)
		print("Offer         :" + offer +"\n\n")
		   

		f.write(brand.replace(",","") + "," +product_name.replace(",","") + "," + retail_price.replace(",","") + "," + price.replace(",","") + ","+ offer.replace(",","") + "\n" )
	f.close()
	print("page " + str(a) + " finished " + "\n")