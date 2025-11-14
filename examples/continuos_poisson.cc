// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) XXXX - YYYY by the polyDEAL authors
//
// This file is part of the polyDEAL library.
//
// Detailed license information governing the source code
// can be found in LICENSE.md at the top level directory.
//
// -----------------------------------------------------------------------------

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>

#define HEX TRUE

struct ConvergenceInfo
{
  ConvergenceInfo() = default;
  void
  add(const std::pair<types::global_dof_index, std::pair<double, double>>
        &dofs_and_errs)
  {
    vec_data.push_back(dofs_and_errs);
  }



  void
  print()
  {
    Assert(vec_data.size() > 0, ExcInternalError());
    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "N DoFs: " << dof_and_errs.first << std::endl;

    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "L2 error: " << dof_and_errs.second.first << std::endl;
    for (const auto &dof_and_errs : vec_data)
      std::cout << std::left << std::setw(24) << std::scientific
                << "H1 error: " << dof_and_errs.second.second << std::endl;
  }



  std::vector<std::pair<types::global_dof_index, std::pair<double, double>>>
    vec_data;
};



enum class GridType
{
  grid_generator, // hyper_cube or hyper_ball
  unstructured    // square generated with gmsh, unstructured
};



enum class PartitionerType
{
  metis,
  rtree,
  no_partition
};



enum SolutionType
{
  linear,      // x+y-1
  quadratic,   // x^2+y^2-1
  product,     // xy(x-1)(y-1)
  product_sine // sin(pi*x)*sin(pi*y)
};



template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const SolutionType &sol_type = SolutionType::linear)
    : Function<dim>()
  {
    solution_type = sol_type;
  }

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  SolutionType solution_type;
};



template <int dim>
void
RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values,
                               const unsigned int /*component*/) const
{
  if (solution_type == SolutionType::linear)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 0.; // Laplacian of linear function
    }
  else if (solution_type == SolutionType::quadratic)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -4.; // quadratic (radial) solution
    }
  else if (solution_type == SolutionType::product)
    {
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = -2. * points[i][0] * (points[i][0] - 1.) -
                    2. * points[i][1] * (points[i][1] - 1.);
    }
  else if (solution_type == SolutionType::product_sine)
    {
      // 2pi^2*sin(pi*x)*sin(pi*y)
      for (unsigned int i = 0; i < values.size(); ++i)
        values[i] = 2. * numbers::PI * numbers::PI *
                    std::sin(numbers::PI * points[i][0]) *
                    std::sin(numbers::PI * points[i][1]);
    }
  else
    {
      Assert(false, ExcNotImplemented());
    }
}



template <int dim>
class SolutionLinear : public Function<dim>
{
public:
  SolutionLinear()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionLinear<dim>::value(const Point<dim> &p, const unsigned int) const
{
  double sum = 0;
  for (unsigned int d = 0; d < dim; ++d)
    sum += p[d];

  return sum - 1; // p[0]+p[1]+p[2]-1
}

template <int dim>
Tensor<1, dim>
SolutionLinear<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  (void)p;
  Tensor<1, dim> return_value;
  for (unsigned int d = 0; d < dim; ++d)
    return_value[d] = 0.;
  return return_value;
}


template <int dim>
void
SolutionLinear<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double>           &values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionQuadratic : public Function<dim>
{
public:
  SolutionQuadratic()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionQuadratic<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * p[0] + p[1] * p[1] - 1; // ball, radial solution
}

template <int dim>
Tensor<1, dim>
SolutionQuadratic<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = 2. * p[0];
  return_value[1] = 2. * p[1];
  return return_value;
}


template <int dim>
void
SolutionQuadratic<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
                                   const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class SolutionProduct : public Function<dim>
{
public:
  SolutionProduct()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;

  virtual void
  gradient_list(const std::vector<Point<dim>> &points,
                std::vector<Tensor<1, dim>>   &gradients,
                const unsigned int /*component*/) const override;
};

template <int dim>
double
SolutionProduct<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return p[0] * (p[0] - 1.) * p[1] * (p[1] - 1.); // square
}

template <int dim>
Tensor<1, dim>
SolutionProduct<dim>::gradient(const Point<dim> &p, const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] = (-1 + 2 * p[0]) * (-1 + p[1]) * p[1];
  return_value[1] = (-1 + 2 * p[1]) * (-1 + p[0]) * p[0];
  return return_value;
}


template <int dim>
void
SolutionProduct<dim>::value_list(const std::vector<Point<dim>> &points,
                                 std::vector<double>           &values,
                                 const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
void
SolutionProduct<dim>::gradient_list(const std::vector<Point<dim>> &points,
                                    std::vector<Tensor<1, dim>>   &gradients,
                                    const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < gradients.size(); ++i)
    gradients[i] = this->gradient(points[i]);
}



template <int dim>
class SolutionProductSine : public Function<dim>
{
public:
  SolutionProductSine()
    : Function<dim>()
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim>  &p,
           const unsigned int component = 0) const override;
};

template <int dim>
double
SolutionProductSine<dim>::value(const Point<dim> &p, const unsigned int) const
{
  return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
}

template <int dim>
Tensor<1, dim>
SolutionProductSine<dim>::gradient(const Point<dim> &p,
                                   const unsigned int) const
{
  Tensor<1, dim> return_value;
  return_value[0] =
    numbers::PI * std::cos(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  return_value[1] =
    numbers::PI * std::cos(numbers::PI * p[1]) * std::sin(numbers::PI * p[0]);
  return return_value;
}


template <int dim>
void
SolutionProductSine<dim>::value_list(const std::vector<Point<dim>> &points,
                                     std::vector<double>           &values,
                                     const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}



template <int dim>
class Poisson
{
private:
  void
  make_grid();
  void
  setup_agglomeration();
  void
  assemble_system();
  void
  solve();
  void
  output_results();


  Triangulation<dim> tria;
#ifdef HEX
  MappingQ1<dim> mapping;
  FE_Q<dim>      fe_q;
#else
  MappingFE<dim>     mapping;
  FE_SimplexDGP<dim> fe_q;
#endif
  std::unique_ptr<AgglomerationHandler<dim>> ah;
  AffineConstraints<double>                  constraints;
  SparsityPattern                            sparsity;
  DynamicSparsityPattern                     dsp;
  SparseMatrix<double>                       system_matrix;
  Vector<double>                             solution;
  Vector<double>                             system_rhs;
  std::unique_ptr<GridTools::Cache<dim>>     cached_tria;
  std::unique_ptr<const Function<dim>>       rhs_function;
  std::unique_ptr<const Function<dim>>       analytical_solution;

public:
  Poisson(const GridType        &grid_type        = GridType::grid_generator,
          const PartitionerType &partitioner_type = PartitionerType::rtree,
          const SolutionType    &solution_type    = SolutionType::linear,
          const unsigned int                      = 0,
          const unsigned int                      = 0,
          const unsigned int fe_degree            = 1);
  void
  run();

  types::global_dof_index
  get_n_dofs() const;

  std::pair<double, double>
  get_error() const;

  GridType        grid_type;
  PartitionerType partitioner_type;
  SolutionType    solution_type;
  unsigned int    extraction_level;
  unsigned int    n_subdomains;
  double penalty_constant = 60.; // 10*(p+1)(p+d) for p = 1 and d = 2 => 60
  double l2_err;
  double semih1_err;
};


template <int dim>
void
create_triangulation_from_bounding_boxes(
  Triangulation<dim>                  &dummy_tria,
  const std::vector<BoundingBox<dim>> &boxes)
{
  const unsigned int n_boxes          = boxes.size();
  const unsigned int vertices_per_box = (dim == 2) ? 4 : 8;

  // Pre-allocate space
  std::vector<Point<dim>>    vertices;
  std::vector<CellData<dim>> cells;
  vertices.reserve(n_boxes * vertices_per_box);
  cells.reserve(n_boxes);

  for (const auto &box : boxes)
    {
      const Point<dim> &p_min = box.get_boundary_points().first;
      const Point<dim> &p_max = box.get_boundary_points().second;

      // Each box gets its own set of vertices (allows overlaps!)
      const unsigned int vertex_offset = vertices.size();

      if constexpr (dim == 2)
        {
          // 4 corners of the bounding box
          vertices.push_back(Point<dim>(p_min[0], p_min[1])); // bottom-left
          vertices.push_back(Point<dim>(p_max[0], p_min[1])); // bottom-right
          vertices.push_back(Point<dim>(p_min[0], p_max[1])); // top-left
          vertices.push_back(Point<dim>(p_max[0], p_max[1])); // top-right

          // Create a quadrilateral cell from these 4 vertices
          CellData<dim> cell;
          cell.vertices[0] = vertex_offset + 0; // bottom-left
          cell.vertices[1] = vertex_offset + 1; // bottom-right
          cell.vertices[2] = vertex_offset + 2; // top-left
          cell.vertices[3] = vertex_offset + 3; // top-right
          cells.push_back(cell);
        }
      else if constexpr (dim == 3)
        {
          // 8 corners of the 3D bounding box
          vertices.push_back(Point<dim>(p_min[0], p_min[1], p_min[2]));
          vertices.push_back(Point<dim>(p_max[0], p_min[1], p_min[2]));
          vertices.push_back(Point<dim>(p_min[0], p_max[1], p_min[2]));
          vertices.push_back(Point<dim>(p_max[0], p_max[1], p_min[2]));
          vertices.push_back(Point<dim>(p_min[0], p_min[1], p_max[2]));
          vertices.push_back(Point<dim>(p_max[0], p_min[1], p_max[2]));
          vertices.push_back(Point<dim>(p_min[0], p_max[1], p_max[2]));
          vertices.push_back(Point<dim>(p_max[0], p_max[1], p_max[2]));

          // Create a hexahedral cell from these 8 vertices
          CellData<dim> cell;
          for (unsigned int v = 0; v < 8; ++v)
            cell.vertices[v] = vertex_offset + v;
          cells.push_back(cell);
        }
    }

  // Create the triangulation
  dummy_tria.create_triangulation(vertices, cells, SubCellData());
}

template <int dim>
void
fill_injection_transfer_matrix(
  const Mapping<dim>            &fine_mapping,
  const DoFHandler<dim>         &coarse_dof_handler, // Questo forse non serve
  const DoFHandler<dim>         &fine_dof_handler,
  const std::vector<Point<dim>> &coarse_support_points,
  const std::vector<Point<dim>> &fine_support_points,
  SparseMatrix<double>          &transfer_matrix,
  SparsityPattern               &sparsity_pattern)
{
  DynamicSparsityPattern dsp(fine_support_points.size(),
                             coarse_support_points.size());

  // Fill sparsity pattern
  for (unsigned int coarse_dof = 0; coarse_dof < coarse_support_points.size();
       ++coarse_dof)
    {
      const Point<dim> &support_point = coarse_support_points[coarse_dof];
      for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
        {
          if (fine_cell->point_inside(support_point))
            {
              std::vector<types::global_dof_index> fine_dof_indices(
                fine_cell->get_fe().dofs_per_cell);
              fine_cell->get_dof_indices(fine_dof_indices);

              for (const auto fine_dof : fine_dof_indices)
                dsp.add(fine_dof, coarse_dof);

              break; // Found the cell, no need to continue
            }
        }
    }

  sparsity_pattern.copy_from(dsp);
  transfer_matrix.reinit(sparsity_pattern);

  for (unsigned int coarse_dof = 0; coarse_dof < coarse_support_points.size();
       ++coarse_dof)
    {
      const Point<dim> &eval_point = coarse_support_points[coarse_dof];

      for (const auto &fine_cell : fine_dof_handler.active_cell_iterators())
        {
          if (fine_cell->point_inside(eval_point))
            {
              // Map physical point to reference coordinates
              Point<dim> ref_point =
                fine_mapping.transform_real_to_unit_cell(fine_cell, eval_point);

              std::vector<types::global_dof_index> fine_dof_indices(
                fine_cell->get_fe().dofs_per_cell);
              fine_cell->get_dof_indices(fine_dof_indices);

              // Evaluate each shape function at the reference point
              for (unsigned int i = 0; i < fine_cell->get_fe().dofs_per_cell;
                   ++i)
                {
                  double shape_value =
                    fine_cell->get_fe().shape_value(i, ref_point);
                  transfer_matrix.set(fine_dof_indices[i],
                                      coarse_dof,
                                      shape_value);
                }

              break; // Found the cell, no need to continue
            }
        }
    }
}

template <int dim>
Poisson<dim>::Poisson(const GridType        &grid_type,
                      const PartitionerType &partitioner_type,
                      const SolutionType    &solution_type,
                      const unsigned int     extraction_level,
                      const unsigned int     n_subdomains,
                      const unsigned int     fe_degree)
  :
#ifdef HEX
  mapping()
#else
  mapping(FE_SimplexP<dim>{1})
#endif
  , fe_q(fe_degree)
  , grid_type(grid_type)
  , partitioner_type(partitioner_type)
  , solution_type(solution_type)
  , extraction_level(extraction_level)
  , n_subdomains(n_subdomains)
  , penalty_constant(10. * (fe_degree + 1) * (fe_degree + dim))
{
  // Initialize manufactured solution
  if (solution_type == SolutionType::linear)
    analytical_solution = std::make_unique<SolutionLinear<dim>>();
  else if (solution_type == SolutionType::quadratic)
    analytical_solution = std::make_unique<SolutionQuadratic<dim>>();
  else if (solution_type == SolutionType::product)
    analytical_solution = std::make_unique<SolutionProduct<dim>>();
  else if (solution_type == SolutionType::product_sine)
    analytical_solution = std::make_unique<SolutionProductSine<dim>>();

  rhs_function = std::make_unique<const RightHandSide<dim>>(solution_type);
  constraints.close();
}

template <int dim>
void
Poisson<dim>::make_grid()
{
  GridIn<dim> grid_in;
  if (grid_type == GridType::unstructured)
    {
      if constexpr (dim == 2)
        {
          grid_in.attach_triangulation(tria);
#ifdef HEX
          std::ifstream gmsh_file(
            "../../meshes/t3.msh"); // unstructured square made by triangles
#else
          std::ifstream gmsh_file(
            "../../meshes/square_simplex_coarser.msh"); // unstructured square
                                                        // made by triangles
#endif
          grid_in.read_msh(gmsh_file);
          tria.refine_global(2); // 4
        }
      else if constexpr (dim == 3)
        {
          grid_in.attach_triangulation(tria);
#ifdef HEX
          std::ifstream filename("../../meshes/piston_3.inp"); // piston mesh
          grid_in.read_abaqus(filename);
#else
          std::ifstream filename(
            "../../meshes/gray_level_image1.vtk"); // liver or brain domain
          grid_in.read_vtk(filename);

#endif
        }
    }
  else
    {
#ifdef HEX
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(3);
#else
      Triangulation<dim> tria_hex;
      GridGenerator::hyper_cube(tria_hex, 0., 1.);
      tria_hex.refine_global(4);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_hex, tria);
#endif
    }
  std::cout << "Size of tria: " << tria.n_active_cells() << std::endl;
  cached_tria = std::make_unique<GridTools::Cache<dim>>(tria, mapping);
  ah          = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

  if (partitioner_type == PartitionerType::metis)
    {
      // Partition the triangulation with graph partitioner.
      auto start = std::chrono::system_clock::now();
      GridTools::partition_triangulation(n_subdomains,
                                         tria,
                                         SparsityTools::Partitioner::metis);

      std::vector<
        std::vector<typename Triangulation<dim>::active_cell_iterator>>
        cells_per_subdomain(n_subdomains);
      for (const auto &cell : tria.active_cell_iterators())
        cells_per_subdomain[cell->subdomain_id()].push_back(cell);

      // For every subdomain, agglomerate elements together
      for (std::size_t i = 0; i < n_subdomains; ++i)
        ah->define_agglomerate(cells_per_subdomain[i]);


      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "METIS built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
    }
  else if (partitioner_type == PartitionerType::rtree)
    {
      // Partition with Rtree

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe_q);

      namespace bgi = boost::geometry::index;
      static constexpr unsigned int max_elem_per_node =
        PolyUtils::constexpr_pow(2, dim); // 2^dim
      std::vector<Point<dim>> support_points_vector(dof_handler.n_dofs());
      unsigned int            i = 0;
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points_vector);

      auto start = std::chrono::system_clock::now();
      auto tree =
        pack_rtree<bgi::rstar<max_elem_per_node>>(support_points_vector);
      std::cout << "Total number of available levels: " << n_levels(tree)
                << std::endl;

#ifdef AGGLO_DEBUG
      // boost::geometry::index::detail::rtree::utilities::print(std::cout,
      // tree);
      Assert(n_levels(tree) >= 2,
             ExcMessage("At least two levels are needed."));
#endif

      CellsAgglomerator<dim, decltype(tree), true> agglomerator{
        tree, extraction_level};
      const auto vec_agglomerates = agglomerator.extract_agglomerates();
      std::cout << "Number of agglomerates: " << vec_agglomerates.size()
                << std::endl;
      // ah->connect_hierarchy(agglomerator);

      // Extracting finest coarse level
      CellsAgglomerator<dim, decltype(tree), true> coarse_agglomerator{
        tree, n_levels(tree) - 1};
      const auto coarse_vec_agglomerates =
        coarse_agglomerator.extract_agglomerates();
      std::cout << "Number of finest agglomerates: "
                << coarse_vec_agglomerates.size() << std::endl;

      std::vector<BoundingBox<dim>> boxes;
      std::vector<BoundingBox<dim>> coarse_boxes;

      for (const auto &agglo : coarse_vec_agglomerates)
        coarse_boxes.emplace_back(agglo);

      for (const auto &agglo : vec_agglomerates)
        {
          boxes.emplace_back(agglo);

          std::cout << "Point in agglomerate: \n";
          for (const auto &point : agglo)
            {
              std::cout << "p: " << point << "\t";
            }
          std::cout << std::endl;
        }
      // ah->define_agglomerate(agglo);

      std::chrono::duration<double> wctduration =
        (std::chrono::system_clock::now() - start);
      std::cout << "R-tree agglomerates built in " << wctduration.count()
                << " seconds [Wall Clock]" << std::endl;
      {
        std::map<types::global_cell_index, types::global_cell_index>
          identity_mapping;
        for (unsigned int j = 0; j < boxes.size(); ++j)
          identity_mapping[j] = j;
        MappingBox<dim> mapping_box(boxes, identity_mapping);

        Triangulation<dim> dummy_tria;
        create_triangulation_from_bounding_boxes(dummy_tria, boxes);

        DoFHandler<dim> support_dof_handler(dummy_tria);
        FE_DGQ<dim>     support_dgfe(fe_q.get_degree());
        support_dof_handler.distribute_dofs(support_dgfe);

        SolutionProductSine<dim> support_function;
        Vector<double>           support_vector(support_dof_handler.n_dofs());
        VectorTools::interpolate(mapping_box,
                                 support_dof_handler,
                                 support_function,
                                 support_vector);

        // Output section
        DataOut<dim> data_out;
        data_out.attach_dof_handler(support_dof_handler);

        data_out.add_data_vector(support_vector, "interpolated_solution");

        Vector<float> cell_indices(dummy_tria.n_active_cells());
        for (const auto &cell : dummy_tria.active_cell_iterators())
          cell_indices[cell->active_cell_index()] = cell->active_cell_index();

        data_out.add_data_vector(cell_indices,
                                 "cell_index",
                                 DataOut<dim>::type_cell_data);

        // Build patches and output
        data_out.build_patches(mapping_box, support_dgfe.get_degree() + 3);
        std::ofstream output("solution_comparison.vtu");
        data_out.write_vtu(output);
      }

      // Let's try and create a transfer matrix with continuous elements
      std::map<types::global_cell_index, types::global_cell_index>
        coarse_identity_mapping;
      for (unsigned int j = 0; j < coarse_boxes.size(); ++j)
        coarse_identity_mapping[j] = j;
      MappingBox<dim> coarse_mapping_box(coarse_boxes, coarse_identity_mapping);

      Triangulation<dim> coarse_bbox_tria;
      create_triangulation_from_bounding_boxes(coarse_bbox_tria, coarse_boxes);

      DoFHandler<dim> coarse_dof_handler(coarse_bbox_tria);
      FE_DGQ<dim>     coarse_dgfe(fe_q.get_degree());
      coarse_dof_handler.distribute_dofs(coarse_dgfe);

      std::vector<Point<dim>> coarse_support_points_vector(
        coarse_dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(coarse_mapping_box,
                                           coarse_dof_handler,
                                           coarse_support_points_vector);
      SparsityPattern      transfer_sp;
      SparseMatrix<double> transfer_matrix;

      fill_injection_transfer_matrix(
        mapping, // Passo mapping perchè devo mappare le celle della griglia
                 // originale!
        coarse_dof_handler,
        dof_handler,
        coarse_support_points_vector,
        support_points_vector,
        transfer_matrix,
        transfer_sp);

      transfer_matrix.print(std::cout);

      std::cout << "Number of coarse support points: "
                << coarse_support_points_vector.size() << std::endl;
      std::cout << "Number of fine support points: "
                << support_points_vector.size() << std::endl;

      std::cout << "Transfer matrix size: " << transfer_matrix.m() << " x "
                << transfer_matrix.n() << std::endl;

      // Check number of agglomerates
      if constexpr (dim == 2)
        {
#ifdef AGGLO_DEBUG
          for (unsigned int j = 0; j < n_subdomains; ++j)
            std::cout << GridTools::count_cells_with_subdomain_association(tria,
                                                                           j)
                      << " cells have subdomain " + std::to_string(j)
                      << std::endl;
#endif
          GridOut           grid_out_svg;
          GridOutFlags::Svg svg_flags;
          svg_flags.background     = GridOutFlags::Svg::Background::transparent;
          svg_flags.line_thickness = 1;
          svg_flags.boundary_line_thickness = 1;
          svg_flags.label_subdomain_id      = true;
          svg_flags.coloring =
            GridOutFlags::Svg::subdomain_id; // GridOutFlags::Svg::none
          grid_out_svg.set_flags(svg_flags);
          std::string   grid_type = "agglomerated_grid";
          std::ofstream out(grid_type + ".svg");
          grid_out_svg.write_svg(tria, out);
        }
    }
  else if (partitioner_type == PartitionerType::no_partition)
    {
    }
  else
    {
      Assert(false, ExcMessage("Wrong partitioning."));
    }
  n_subdomains = ah->n_agglomerates();
  std::cout << "N subdomains = " << n_subdomains << std::endl;
}

template <int dim>
void
Poisson<dim>::setup_agglomeration()
{
  if (partitioner_type == PartitionerType::no_partition)
    {
      // No partitioning means that each cell is a master cell
      for (const auto &cell : tria.active_cell_iterators())
        ah->define_agglomerate({cell});
    }

  ah->distribute_agglomerated_dofs(
    fe_q); // Qui c'è da rifare un po' tutto. Stavolta non uso dg ma standard
           // lagrangian elements
  ah->create_agglomeration_sparsity_pattern(dsp);
  sparsity.copy_from(dsp);

  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename =
      "grid" + partitioner + "_" + std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ah->agglo_dh);

    Vector<float> agglomerated(tria.n_active_cells());
    Vector<float> agglo_idx(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      {
        agglomerated[cell->active_cell_index()] =
          ah->get_relationships()[cell->active_cell_index()];
        agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
      }
    data_out.add_data_vector(agglomerated,
                             "agglo_relationships",
                             DataOut<dim>::type_cell_data);
    data_out.add_data_vector(agglo_idx,
                             "agglomerated_idx",
                             DataOut<dim>::type_cell_data);
    data_out.build_patches(mapping);
    data_out.write_vtu(output);
  }
}



template <int dim>
void
Poisson<dim>::assemble_system()
{
  system_matrix.reinit(sparsity);
  solution.reinit(ah->n_dofs());
  system_rhs.reinit(ah->n_dofs());

  const unsigned int quadrature_degree      = fe_q.get_degree() + 1;
  const unsigned int face_quadrature_degree = fe_q.get_degree() + 1;
#ifdef HEX
  // Questi penso vanno bene così, non ho capito a pieno
  // i dettagli dell'implementazione che c'è dietro
  ah->initialize_fe_values(QGauss<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGauss<dim - 1>(face_quadrature_degree));
#else
  ah->initialize_fe_values(QGaussSimplex<dim>(quadrature_degree),
                           update_gradients | update_JxW_values |
                             update_quadrature_points | update_JxW_values |
                             update_values,
                           QGaussSimplex<dim - 1>(face_quadrature_degree));
#endif

  const unsigned int dofs_per_cell = ah->n_dofs_per_cell();
  std::cout << "DoFs per cell: " << dofs_per_cell << std::endl;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // Next, we define the four dofsxdofs matrices needed to assemble jumps and
  // averages. Questi posso togliere tutto, in teoria se uso elementi continui
  // non servono salti e medie
  FullMatrix<double> M11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M12(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M21(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> M22(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &polytope : ah->polytope_iterators())
    { // Qui anche non penso ci sia moltissimo da cambiare, dati i politopi
      // ottengo i punti di quadratura. Chiarito come si agglomerano i politopi
      // qui non dovrei toccar nulla
#ifdef AGGLO_DEBUG
      std::cout << "Polytope with idx: " << polytope->index() << std::endl;
#endif
      cell_matrix              = 0;
      cell_rhs                 = 0;
      const auto &agglo_values = ah->reinit(polytope);
      polytope->get_dof_indices(local_dof_indices);

      const auto         &q_points  = agglo_values.get_quadrature_points();
      const unsigned int  n_qpoints = q_points.size();
      std::vector<double> rhs(n_qpoints);
      rhs_function->value_list(q_points, rhs);

      for (unsigned int q_index : agglo_values.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += agglo_values.shape_grad(i, q_index) *
                                       agglo_values.shape_grad(j, q_index) *
                                       agglo_values.JxW(q_index);
                }
              cell_rhs(i) += agglo_values.shape_value(i, q_index) *
                             rhs[q_index] * agglo_values.JxW(q_index);
            }
        }


      // Face terms
      const unsigned int n_faces = polytope->n_faces();
      AssertThrow(n_faces > 0,
                  ExcMessage(
                    "Invalid element: at least 4 faces are required."));


#ifdef AGGLO_DEBUG
      std::cout << "Face loop for " << polytope->index() << std::endl;
      std::cout << "n faces = " << n_faces << std::endl;
#endif

      auto polygon_boundary_vertices = polytope->polytope_boundary();
      for (unsigned int f = 0; f < n_faces; ++f)
        {
          // Qui se ho capito bene sta imponendo le condizioni al bordo con
          // penalizzazione
          if (polytope->at_boundary(f))
            {
              // std::cout << "at boundary!" << std::endl;
              const auto &fe_face = ah->reinit(polytope, f);

              const unsigned int dofs_per_cell = fe_face.dofs_per_cell;
              // std::cout << "With dofs_per_cell =" << fe_face.dofs_per_cell
              //           << std::endl;

              const auto &face_q_points = fe_face.get_quadrature_points();
              std::vector<double> analytical_solution_values(
                face_q_points.size());
              analytical_solution->value_list(face_q_points,
                                              analytical_solution_values,
                                              1);

              // Get normal vectors seen from each agglomeration.
              const auto &normals = fe_face.get_normal_vectors();

              // const double penalty =
              //   penalty_constant / PolyUtils::compute_h_orthogonal(
              //                        f, polygon_boundary_vertices,
              //                        normals[0]);

              const double penalty =
                penalty_constant / std::fabs(polytope->diameter());

              for (unsigned int q_index : fe_face.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) +=
                            (-fe_face.shape_value(i, q_index) *
                               fe_face.shape_grad(j, q_index) *
                               normals[q_index] -
                             fe_face.shape_grad(i, q_index) * normals[q_index] *
                               fe_face.shape_value(j, q_index) +
                             (penalty)*fe_face.shape_value(i, q_index) *
                               fe_face.shape_value(j, q_index)) *
                            fe_face.JxW(q_index);
                        }
                      cell_rhs(i) +=
                        (penalty * analytical_solution_values[q_index] *
                           fe_face.shape_value(i, q_index) -
                         fe_face.shape_grad(i, q_index) * normals[q_index] *
                           analytical_solution_values[q_index]) *
                        fe_face.JxW(q_index);
                    }
                }
            }
          else
            {
              const auto &neigh_polytope = polytope->neighbor(f);
#ifdef AGGLO_DEBUG
              std::cout << "Neighbor is " << neigh_polytope->index()
                        << std::endl;
#endif


              // This is necessary to loop over internal faces only once.
              if (polytope->index() < neigh_polytope->index())
                {
                  unsigned int nofn =
                    polytope->neighbor_of_agglomerated_neighbor(f);
#ifdef AGGLO_DEBUG
                  std::cout << "Neighbor of neighbor is:" << nofn << std::endl;
#endif
                  const auto &fe_faces =
                    ah->reinit_interface(polytope, neigh_polytope, f, nofn);
#ifdef AGGLO_DEBUG
                  std::cout << "Reinited the interface:" << nofn << std::endl;
#endif
                  const auto &fe_faces0 = fe_faces.first;
                  const auto &fe_faces1 = fe_faces.second;

#ifdef AGGLO_DEBUG
                  std::cout << "Local from current: " << f << std::endl;
                  std::cout << "Local from neighbor: " << nofn << std::endl;

                  std::cout << "Jump between " << polytope->index() << " and "
                            << neigh_polytope->index() << std::endl;
                  {
                    std::cout << "Quadrature points from first polytope: "
                              << std::endl;
                    for (const auto &q : fe_faces0.get_quadrature_points())
                      std::cout << q << std::endl;
                    std::cout << "Quadrature points from second polytope: "
                              << std::endl;
                    for (const auto &q : fe_faces1.get_quadrature_points())
                      std::cout << q << std::endl;


                    std::cout << "Check: " << std::endl;
                    const auto &points0 = fe_faces0.get_quadrature_points();
                    const auto &points1 = fe_faces1.get_quadrature_points();
                    for (size_t i = 0;
                         i < fe_faces1.get_quadrature_points().size();
                         ++i)
                      {
                        double d = (points0[i] - points1[i]).norm();
                        AssertThrow(
                          d < 1e-15,
                          ExcMessage(
                            "Face qpoints at the interface do not match!"));
                        std::cout << d << std::endl;
                      }
                  }
#endif

                  std::vector<types::global_dof_index>
                    local_dof_indices_neighbor(dofs_per_cell);

                  M11 = 0.;
                  M12 = 0.;
                  M21 = 0.;
                  M22 = 0.;

                  const auto &normals = fe_faces0.get_normal_vectors();

                  const double penalty =
                    penalty_constant / std::fabs(polytope->diameter());

                  // M11
                  for (unsigned int q_index :
                       fe_faces0.quadrature_point_indices())
                    {
#ifdef AGGLO_DEBUG
                      std::cout << normals[q_index] << std::endl;
#endif
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              M11(i, j) +=
                                (-0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) -
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) +
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces0.JxW(q_index);

                              M12(i, j) +=
                                (0.5 * fe_faces0.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) -
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(i, q_index) -
                                 (penalty)*fe_faces0.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              // A10
                              M21(i, j) +=
                                (-0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces0.shape_value(j, q_index) +
                                 0.5 * fe_faces0.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) -
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
                                   fe_faces0.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);

                              // A11
                              M22(i, j) +=
                                (0.5 * fe_faces1.shape_grad(i, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(j, q_index) +
                                 0.5 * fe_faces1.shape_grad(j, q_index) *
                                   normals[q_index] *
                                   fe_faces1.shape_value(i, q_index) +
                                 (penalty)*fe_faces1.shape_value(i, q_index) *
                                   fe_faces1.shape_value(j, q_index)) *
                                fe_faces1.JxW(q_index);
                            }
                        }
                    }

                  neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

                  constraints.distribute_local_to_global(M11,
                                                         local_dof_indices,
                                                         system_matrix);
                  constraints.distribute_local_to_global(
                    M12,
                    local_dof_indices,
                    local_dof_indices_neighbor,
                    system_matrix);
                  constraints.distribute_local_to_global(
                    M21,
                    local_dof_indices_neighbor,
                    local_dof_indices,
                    system_matrix);
                  constraints.distribute_local_to_global(
                    M22, local_dof_indices_neighbor, system_matrix);
                } // Loop only once trough internal faces
            }
        } // Loop over faces of current cell

      // distribute DoFs
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    } // Loop over cells
}



template <int dim>
void
Poisson<dim>::solve()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);
}



template <int dim>
void
Poisson<dim>::output_results()
{
  {
    std::string partitioner;
    if (partitioner_type == PartitionerType::metis)
      partitioner = "metis";
    else if (partitioner_type == PartitionerType::rtree)
      partitioner = "rtree";
    else
      partitioner = "no_partitioning";

    const std::string filename = "interpolated_solution" + partitioner + "_" +
                                 std::to_string(n_subdomains) + ".vtu";
    std::ofstream output(filename);

    DataOut<dim>   data_out;
    Vector<double> interpolated_solution;
    PolyUtils::interpolate_to_fine_grid(*ah,
                                        interpolated_solution,
                                        solution,
                                        true /*on_the_fly*/);
    data_out.attach_dof_handler(ah->output_dh);
    data_out.add_data_vector(interpolated_solution,
                             "u",
                             DataOut<dim>::type_dof_data);

    Vector<float> agglo_idx(tria.n_active_cells());

    // Mark fine cells belonging to the same agglomerate.
    for (const auto &polytope : ah->polytope_iterators())
      {
        const types::global_cell_index polytope_index = polytope->index();
        const auto &patch_of_cells = polytope->get_agglomerate(); // fine cells
        // Flag them
        for (const auto &cell : patch_of_cells)
          agglo_idx[cell->active_cell_index()] = polytope_index;
      }

    // Old way, here just for completeness
    // for (const auto &cell : tria.active_cell_iterators())
    // {
    //   agglomerated[cell->active_cell_index()] =
    //     ah->get_relationships()[cell->active_cell_index()];
    //   agglo_idx[cell->active_cell_index()] = cell->subdomain_id();
    // }

    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(mapping);
    data_out.write_vtu(output);

    // Compute L2 and semiH1 norm of error
    std::vector<double> errors;
    PolyUtils::compute_global_error(*ah,
                                    solution,
                                    *analytical_solution,
                                    {VectorTools::L2_norm,
                                     VectorTools::H1_seminorm},
                                    errors);
    l2_err     = errors[0];
    semih1_err = errors[1];
    std::cout << "Error (L2): " << l2_err << std::endl;
    std::cout << "Error (H1): " << semih1_err << std::endl;
  }
}



template <int dim>
inline types::global_dof_index
Poisson<dim>::get_n_dofs() const
{
  return ah->n_dofs();
}



template <int dim>
inline std::pair<double, double>
Poisson<dim>::get_error() const
{
  return std::make_pair(l2_err, semih1_err);
}



template <int dim>
void
Poisson<dim>::run()
{
  make_grid();
  setup_agglomeration();
  auto start = std::chrono::high_resolution_clock::now();
  assemble_system();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Time taken by assemble_system(): " << duration.count() / 1e6
            << " seconds" << std::endl;
  solve();
  output_results();
}



int
main()
{
  // Testing p-convergence
  ConvergenceInfo convergence_info;
  std::cout << "Testing p-convergence" << std::endl;
  {
#ifdef HEX
    for (unsigned int fe_degree : {1, 2, 3, 4})
#else
    for (unsigned int fe_degree : {1, 2, 3})
#endif
      {
        std::cout << "Fe degree: " << fe_degree << std::endl;
        Poisson<2> poisson_problem{GridType::grid_generator,
                                   PartitionerType::rtree,
                                   SolutionType::product_sine,
                                   2 /*extraction_level*/,
                                   0,
                                   fe_degree};
        poisson_problem.run();
        convergence_info.add(
          std::make_pair<types::global_dof_index, std::pair<double, double>>(
            poisson_problem.get_n_dofs(), poisson_problem.get_error()));
      }
  }
  convergence_info.print();


  std::cout << std::endl;
  return 0;
}
