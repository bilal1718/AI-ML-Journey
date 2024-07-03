distanceOfTripInKilometers=int(input("Enter the distance of trip in kilometers: "))
fuelEffiencyOfCar=float(input("Enter the fuel effiency of car (kilometers per liter) : "))
costPerLiterOfFuel=float(input("Enter the cost per liter of fuel: "))

totalFuelRequiredForTrip=distanceOfTripInKilometers * fuelEffiencyOfCar
totalCostRequiredForTrip=totalFuelRequiredForTrip * costPerLiterOfFuel
print("The total fuel required for the trip is: ", totalFuelRequiredForTrip)
print("The total cost of the trip based on the fuel efficiency and cost per liter is: ", totalCostRequiredForTrip)