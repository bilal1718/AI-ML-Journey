import './App.css'
import { useState } from 'react';
import axios from 'axios';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [formData, setFormData] = useState({
    BuildingType: '',
    PrimaryPropertyType: '',
    YearBuilt: '',
    NumberofBuildings: '',
    NumberofFloors: '',
    PropertyGFATotal: '',
    PropertyGFAParking: '',
    PropertyGFABuilding: '',
    SiteEnergyUse: '',
    Electricity: '',
    NaturalGas: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const numericData = {
      ...formData,
      YearBuilt: Number(formData.YearBuilt),
      NumberofBuildings: Number(formData.NumberofBuildings),
      NumberofFloors: Number(formData.NumberofFloors),
      PropertyGFATotal: Number(formData.PropertyGFATotal),
      PropertyGFAParking: Number(formData.PropertyGFAParking),
      PropertyGFABuilding: Number(formData.PropertyGFABuilding),
      SiteEnergyUse: Number(formData.SiteEnergyUse),
      Electricity: Number(formData.Electricity),
      NaturalGas: Number(formData.NaturalGas)
    };

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', numericData);
      console.log('Prediction:', response.data.prediction);
      setPrediction(response.data.prediction)
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };
  return (
    <>
      <link
        rel="stylesheet"
        href="https://demos.creative-tim.com/notus-js/assets/styles/tailwind.css"
      />
      <link
        rel="stylesheet"
        href="https://demos.creative-tim.com/notus-js/assets/vendor/@fortawesome/fontawesome-free/css/all.min.css"
      />
      <section className="py-1 bg-blueGray-50">
        <div className="w-full lg:w-8/12 px-4 mx-auto mt-6">
          <div className="relative flex flex-col min-w-0 break-words w-full mb-6 shadow-lg rounded-lg bg-blueGray-100 border-0">
            <div className="rounded-t bg-white mb-0 px-6 py-6">
              <div className="text-center flex justify-between">
                <h6 className="text-blueGray-700 text-2xl font-bold">Site Energy Use Intensity AI Model</h6>
              </div>
            </div>
            <div className="flex-auto px-4 lg:px-10 py-10 pt-0">
              <form onSubmit={handleSubmit}>
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Building Details
                </h6>
                <div className="flex flex-wrap">
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NumberofBuildings"
                      >
                        Number of Buildings
                      </label>
                      <input
                        type="number"
                        name="NumberofBuildings"
                        id="NumberofBuildings"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter number of buildings"
                        value={formData.NumberofBuildings}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NumberofFloors"
                      >
                        Number of Floors
                      </label>
                      <input
                        type="number"
                        name="NumberofFloors"
                        id="NumberofFloors"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter number of floors"
                        value={formData.NumberofFloors}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="Electricity"
                      >
                        Electricity (kWh)
                      </label>
                      <input
                        type="number"
                        name="Electricity"
                        id="Electricity"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter electricity usage"
                        value={formData.Electricity}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NaturalGas"
                      >
                        Natural Gas (therms)
                      </label>
                      <input
                        type="number"
                        name="NaturalGas"
                        id="NaturalGas"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter natural gas usage"
                        value={formData.NaturalGas}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <hr className="mt-6 border-b-1 border-blueGray-300" />
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Property Details
                </h6>
                <div className="flex flex-wrap">
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="BuildingType_Encoded"
                      >
                        Building Type
                      </label>
                      <input
                        type="number"
                        name="BuildingType_Encoded"
                        id="BuildingType_Encoded"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter building type"
                        value={formData.BuildingType_Encoded}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PrimaryPropertyType_Encoded"
                      >
                        Primary Property Type
                      </label>
                      <input
                        type="number"
                        name="PrimaryPropertyType_Encoded"
                        id="PrimaryPropertyType_Encoded"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter property type"
                        value={formData.PrimaryPropertyType_Encoded}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="YearBuilt"
                      >
                        Year Built
                      </label>
                      <input
                        type="number"
                        name="YearBuilt"
                        id="YearBuilt"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter year built"
                        value={formData.YearBuilt}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <hr className="mt-6 border-b-1 border-blueGray-300" />
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Area Details
                </h6>
                <div className="flex flex-wrap">
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFATotal"
                      >
                        Property GFA Total
                      </label>
                      <input
                        type="number"
                        name="PropertyGFATotal"
                        id="PropertyGFATotal"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter total GFA"
                        value={formData.PropertyGFATotal}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFAParking"
                      >
                        Property GFA Parking
                      </label>
                      <input
                        type="number"
                        name="PropertyGFAParking"
                        id="PropertyGFAParking"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter GFA for parking"
                        value={formData.PropertyGFAParking}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFABuilding"
                      >
                        Property GFA Building
                      </label>
                      <input
                        type="number"
                        name="PropertyGFABuilding"
                        id="PropertyGFABuilding"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter GFA for building"
                        value={formData.PropertyGFABuilding}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="SiteEnergyUse"
                      >
                        Site Energy Use
                      </label>
                      <input
                        type="number"
                        name="SiteEnergyUse"
                        id="SiteEnergyUse"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter site energy use"
                        value={formData.SiteEnergyUse}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <div className="flex justify-end mt-6">
                  <button
                    type="submit"
                    className="bg-blueGray-600 text-white active:bg-blueGray-700 text-sm font-bold uppercase px-6 py-3 rounded shadow hover:shadow-lg outline-none focus:outline-none mr-1 ease-linear transition-all duration-150"
                  >
                    Submit
                  </button>
                </div>
              </form>
              {prediction && (
                <div className="mt-6 p-4 bg-white rounded shadow-md">
                  <h3 className="text-lg font-bold">Prediction</h3>
                  <p className="mt-2 text-blueGray-600">{prediction}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

export default App;
