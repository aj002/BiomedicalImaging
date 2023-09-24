import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatismoIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel}
import scalismo.registration._
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object Hello extends App {
import scalismo.mesh.TriangleMesh
scalismo.initialize()
implicit val rng = scalismo.utils.Random(42)
val ui = ScalismoUI()
//load and display reference femur
val referenceFemur: TriangleMesh[_3D] = MeshIO.readMesh(new
java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/datasets/femur.stl")).get
val femurGroup = ui.createGroup("femur")
//val referenceView = ui.show(femurGroup, referenceFemur, "Femur")
//building shape model from analytically defined mean and covariance

val zeroMean = VectorField(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D]) val kernel = DiagonalKernel[_3D](GaussianKernel(sigma = 130) * 0.001, outputDim = 3)
val refgp = GaussianProcess(zeroMean, kernel)
val reflowRankGP = LowRankGaussianProcess.approximateGPCholesky(
referenceFemur.pointSet,
refgp,
relativeTolerance = 0.05,
interpolator = NearestNeighborInterpolator()
)
val gpView = ui.addTransformation(femurGroup, reflowRankGP, "gp") val defField: Field[_3D, EuclideanVector[_3D]] = reflowRankGP.sample referenceFemur.transform((p: Point[_3D]) => p + defField(p))
val ssm = StatisticalMeshModel(referenceFemur, reflowRankGP)
//val ssmView = ui.show(femurGroup, ssm, "group")
//val gpView = ui.addTransformation(femurGroup, lowRankGP, "gp") //saving ssm
StatismoIO.writeStatismoMeshModel(ssm, new
java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/datasets/GenericSSMFemur.h5")).get

//loading and displaying from femur data
val meshFiles = new java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/Courses_SSM2016_Training_step2/step2/meshes").listFiles
val dsGroup = ui.createGroup("datasets")
/*val (meshes, meshViews) = meshFiles.map(meshFile => {
val mesh = MeshIO.readMesh(meshFile).get
val meshView = ui.show(dsGroup, mesh, "mesh") (mesh, meshView)
}) .unzip*/
val meshes = meshFiles.map(meshFile => { val mesh = MeshIO.readMesh(meshFile).get
mesh })
//rigid alignment of data to reference femur
val landmarkFiles = new java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/Courses_SSM2016_Training_step2/step2/landmarks").listFiles
val landmarks = landmarkFiles.map { id => LandmarkIO.readLandmarksJson[_3D](id).get } val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/datasets/femur.json")).get


al alignedTransform = landmarks.map { lm =>
val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(lm, referenceLandmarks, center =
Point(0, 0, 0)) rigidTrans
}
val alignedSet = (0 until 50).map { d =>
val align = meshes(d).transform(alignedTransform(d)) //ui.show(align,"Aligned_"+d)
align
}
//saving aligned Meshes
/*val sampleMeshDir = "C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/Courses_SSM2016_Training_step2/step2/aligned_meshes/"
(0 until 50).foreach(i => MeshIO.writeMesh(alignedSet(i), new java.io.File(sampleMeshDir + s"alignedMesh$i.stl")))*/

al alignedMeshFiles = new java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/Courses_SSM2016_Training_step2/step2/aligned_meshes").listFiles()
val targetMeshes: Array[TriangleMesh[_3D]] = alignedMeshFiles.map(meshFile => MeshIO.readMesh(meshFile).get)
val refMesh = referenceFemur
val refMeshGroup = ui.createGroup("refMeshGroup")
val refMeshView = ui.show(refMeshGroup, refMesh, "refMesh") refMeshView.color = java.awt.Color.GREEN
case class RegistrationParameters(regularizationWeight: Double, numberOfIterations: Int, numberOfSampledPoints: Int)
def doRegistration(
lowRankGP: LowRankGaussianProcess[_3D, EuclideanVector[_3D]], refMesh: TriangleMesh[_3D],
targetMesh: TriangleMesh[_3D],
registrationParameters: RegistrationParameters, initialCoefficients: DenseVector[Double]
): DenseVector[Double] = {
val transformationSpace = GaussianProcessTransformationSpace(lowRankGP) val fixedImage = refMesh.operations.toDistanceImage
val movingImage = targetMesh.operations.toDistanceImage
val sampler = FixedPointsUniformMeshSampler3D(refMesh, registrationParameters.numberOfSampledPoints)
val metric = MeanSquaresMetric(fixedImage, movingImage, transformationSpace, sampler) val optimizer = LBFGSOptimizer(registrationParameters.numberOfIterations)
val regularizer = L2Regularizer(transformationSpace)
val registration = Registration(
metric,
regularizer, registrationParameters.regularizationWeight, optimizer
)
val registrationIterator = registration.iterator(initialCoefficients)
val visualizingRegistrationIterator = for ((it, itnum) <- registrationIterator.zipWithIndex) yield {
println(s"object value in iteration $itnum is ${it.value}") //gpView.coefficients = it.parameters
it
}
val registrationResult = visualizingRegistrationIterator.toSeq.last registrationResult.parameters
}
val refLMs = referenceLandmarks
val refLMNodeIds = refLMs.map(eachLM => refMesh.pointSet.findClosestPoint(eachLM.point).id)
val knownDeformations = (0 until 50).map(i => {
landmarks(0).indices.map(j => landmarks(i + 1)(j).point - refMesh.pointSet.point(refLMNodeIds(j)))
})
val refLMpoints = refLMNodeIds.toIndexedSeq.map(eachId => refMesh.pointSet.point(eachId)) val refLMdomain = UnstructuredPointsDomain(refLMpoints)
val knownDeformationFields = knownDeformations.map(eachDefSet => {
DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](refLMdomain, eachDefSet)
})
val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), 0.01 * DenseMatrix.eye[Double](3))

ef RegressionDataGenerator(dataPoints: IndexedSeq[Point[_3D]], deformations: IndexedSeq[EuclideanVector[_3D]], noise: MultivariateNormalDistribution) = {
val regressionDataSet = dataPoints.indices.map(i => (dataPoints(i), deformations(i), noise))regressionDataSet }
val regressionDataSets = (0 until 50).map(i => RegressionDataGenerator(refLMpoints, knownDeformations(i), noise))
for (n <- 0 until 2) {
val posteriorGP = reflowRankGP.posterior(regressionDataSets(n))
val initialCoefficients = DenseVector.zeros[Double](posteriorGP.rank) val registrationParameters = Seq(
RegistrationParameters(regularizationWeight = 1e-1, numberOfIterations = 20, numberOfSampledPoints = 1000),
RegistrationParameters(regularizationWeight = 1e-2, numberOfIterations = 30, numberOfSampledPoints = 1000),
RegistrationParameters(regularizationWeight = 1e-4, numberOfIterations = 40, numberOfSampledPoints = 2000),
RegistrationParameters(regularizationWeight = 1e-6, numberOfIterations = 50,
numberOfSampledPoints = 4000) )
val finalCoefficients = registrationParameters.foldLeft(initialCoefficients)((modelCoefficients, regParameters) =>
doRegistration(posteriorGP, refMesh, targetMeshes(n), regParameters, modelCoefficients))
val transformationSpace = GaussianProcessTransformationSpace(posteriorGP)
val registrationTransformation = transformationSpace.transformForParameters(finalCoefficients) val fittedMesh = refMesh.transform(registrationTransformation)
val fittedMeshGroup = ui.createGroup("fittedMeshGroup")
ui.show(fittedMeshGroup, fittedMesh, "fittedMesh")
ui.show(fittedMeshGroup, targetMeshes(n), s"TargetMesh $n")
val
registeredMeshDir="C:/Users/Home/Desktop/jjn/internship/Courses_SSM2016_Training_step2/step2/r egistered_meshes/"
val fittedMeshFile = new java.io.File(registeredMeshDir + s"ftdMesh$n.stl") MeshIO.writeSTL(fittedMesh, fittedMeshFile)
}


/building shape model using PCA
val registeredMeshFiles = new java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/Courses_SSM2016_Training_step2/step2/registered_meshes").listFiles()
val registeredMeshes: Array[TriangleMesh[_3D]] = registeredMeshFiles.map(meshFile => MeshIO.readMesh(meshFile).get)
val defFields = registeredMeshes.map{ m =>
val deformationVectors = referenceFemur.pointSet.pointIds.map{ id : PointId => m.pointSet.point(id) - referenceFemur.pointSet.point(id)
}.toIndexedSeq
DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](referenceFemur.pointSet, deformationVectors)
}
val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
val continuousFields = defFields.map(f => f.interpolate(interpolator) )
val newgp = DiscreteLowRankGaussianProcess.createUsingPCA(referenceFemur.pointSet,
continuousFields)

val newmodel = StatisticalMeshModel(referenceFemur, newgp.interpolate(interpolator)) val modelGroup = ui.createGroup("newmodel")
val newssmView = ui.show(modelGroup, newmodel, "newmodel") StatismoIO.writeStatismoMeshModel(newmodel, new
java.io.File("C:/Users/Home/Desktop/jjn/Scalismo/scalismolab- win64/scalismolab/datasets/NewSSMFemur.h5")).get
}
