SELECT ?organisation ?organisationLabel ?industryLabel ?midLabel ?url
WHERE {
  
  ?organisation wdt:P31 wd:Q4830453 .
  OPTIONAL {?organisation wdt:P17 ?country}
  OPTIONAL {?organisation wdt:P159 ?hq}
  OPTIONAL {?organisation wdt:P452 ?industry}
  
  OPTIONAL {?organisation wdt:P646 ?mid}
  OPTIONAL {?organisation wdt:P414 ?se}
  OPTIONAL {?organisation wdt:P249 ?ticker}
  OPTIONAL {?organisation wdt:P856 ?url}
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
}          
                
} 
